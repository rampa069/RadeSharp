// SPDX-License-Identifier: BSD-2-Clause
// Command-line equivalents of the rade_c tools, same arguments and stream formats:
//   radesharp radae_tx [--v2]              features.f32 (stdin) -> IQ f32 (stdout)
//   radesharp radae_rx [--v2] [-v N] [--agc 0|1]  IQ (stdin) -> features (stdout)
//   radesharp lpcnet_demo -features|-fargan-synthesis <in> <out>
//   radesharp rade_tx_wav [--v2] [--no_bpf] [-v N] [-f feat.f32] <speech16k.wav> <out8k.wav>
//   radesharp rade_rx_wav [--v2] [-v N] [-f feat.f32] <in8k.wav> <speech16k.wav>

using System.Globalization;
using System.Runtime.InteropServices;
using RadeSharp;
using RadeSharp.Tools;
using RadeSharp.Vocoder;
using static RadeSharp.RadeApi;

if (args.Length == 0) return Usage();
var rest = args[1..];
return args[0] switch
{
    "radae_tx" => RadaeTx(rest),
    "radae_rx" => RadaeRx(rest),
    "lpcnet_demo" => LpcnetDemo(rest),
    "rade_tx_wav" => RadeTxWav(rest),
    "rade_rx_wav" => RadeRxWav(rest),
    _ => Usage(),
};

static int Usage()
{
    Console.Error.WriteLine("usage: radesharp <radae_tx|radae_rx|lpcnet_demo|rade_tx_wav|rade_rx_wav> [args]");
    return 1;
}

static (Dictionary<string, string?> opts, List<string> pos) Parse(string[] a, params string[] withValue)
{
    var opts = new Dictionary<string, string?>();
    var pos = new List<string>();
    for (int i = 0; i < a.Length; i++)
    {
        if (a[i].StartsWith('-') && a[i] != "-" && !(a[i].Length > 1 && char.IsDigit(a[i][1])))
        {
            string k = a[i].TrimStart('-');
            opts[k] = withValue.Contains(k) && i + 1 < a.Length ? a[++i] : null;
        }
        else pos.Add(a[i]);
    }
    return (opts, pos);
}

static int ReadExact(Stream s, Span<byte> buf)
{
    int total = 0;
    while (total < buf.Length)
    {
        int n = s.Read(buf[total..]);
        if (n == 0) break;
        total += n;
    }
    return total;
}

static int RadaeTx(string[] a)
{
    var (o, _) = Parse(a, "m", "model_name");
    int flags = o.ContainsKey("v2") ? RADE_MODE_V2 : 0;
    var r = rade_open("(unused, built-in weights)", flags)!;
    int nf = rade_n_features_in_out(r), ntx = rade_n_tx_out(r), neoo = rade_n_tx_eoo_out(r);
    if (File.Exists("eoo_tx.f32"))
    {
        Console.Error.WriteLine("Setting EOO");
        var bits = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes("eoo_tx.f32"));
        rade_tx_set_eoo_bits(r, bits[..rade_n_eoo_bits(r)]);
    }
    Console.Error.WriteLine($"n_features_in: {nf} n_tx_out: {ntx} n_eoo_out: {neoo}");
    using var stdin = Console.OpenStandardInput();
    using var stdout = Console.OpenStandardOutput();
    var feat = new float[nf];
    var txOut = new RadeComp[Math.Max(ntx, neoo)];
    int frames = 0;
    while (ReadExact(stdin, MemoryMarshal.AsBytes(feat.AsSpan())) == nf * 4)
    {
        int n = rade_tx(r, txOut, feat);
        stdout.Write(MemoryMarshal.AsBytes(txOut.AsSpan(0, n)));
        frames++;
    }
    int ne = rade_tx_eoo(r, txOut);
    stdout.Write(MemoryMarshal.AsBytes(txOut.AsSpan(0, ne)));
    stdout.Write(new byte[ne * 8]);   // extra silence to let Rx finish processing EOO
    Console.Error.WriteLine($"Transmitted {frames} modem frames + EOO");
    rade_close(r);
    return 0;
}

static int RadaeRx(string[] a)
{
    var (o, _) = Parse(a, "v", "m", "model_name", "disable_unsync", "write_snr_est", "gain", "agc");
    int flags = 0;
    if (o.TryGetValue("v", out var vs))
    {
        int v = int.Parse(vs!, CultureInfo.InvariantCulture);
        if (v == 0) flags |= RADE_VERBOSE_0;
        else if (v == 2) flags |= RADE_VERBOSE_TERSE;
        else if (v >= 3) flags |= RADE_VERBOSE_FULL;
    }
    if (o.ContainsKey("v2")) flags |= RADE_MODE_V2;
    var r = rade_open("(unused, built-in weights)", flags)!;
    float gain = o.TryGetValue("gain", out var g) ? float.Parse(g!, CultureInfo.InvariantCulture) : 1.0f;
    if (o.TryGetValue("disable_unsync", out var du)) rade_set_disable_unsync(r, float.Parse(du!, CultureInfo.InvariantCulture));
    if (o.TryGetValue("agc", out var agc)) rade_rx_set_agc(r, int.Parse(agc!, CultureInfo.InvariantCulture));
    int ninMax = rade_nin_max(r), nf = rade_n_features_in_out(r), nbits = rade_n_eoo_bits(r);
    Console.Error.WriteLine($"nin_max: {ninMax} n_features_out: {nf} n_eoo_bits: {nbits}");
    using var stdin = Console.OpenStandardInput();
    using var stdout = Console.OpenStandardOutput();
    using var feoo = File.Create("eoo_rx.f32");
    var snrLog = new List<float>();
    var rxIn = new RadeComp[ninMax];
    var feat = new float[nf];
    var eoo = new float[nbits];
    int syms = 0, valid = 0;
    while (true)
    {
        int nin = rade_nin(r);
        if (ReadExact(stdin, MemoryMarshal.AsBytes(rxIn.AsSpan(0, nin))) != nin * 8) break;
        if (gain != 1.0f) for (int i = 0; i < nin; i++) rxIn[i] = new(rxIn[i].Real * gain, rxIn[i].Imag * gain);
        int nout = rade_rx(r, feat, out int hasEoo, eoo, rxIn.AsSpan(0, nin));
        if (nout > 0)
        {
            stdout.Write(MemoryMarshal.AsBytes(feat.AsSpan(0, nout)));
            valid++;
        }
        if (hasEoo != 0)
        {
            Console.Error.WriteLine("End-of-over detected");
            feoo.Write(MemoryMarshal.AsBytes(eoo.AsSpan()));
        }
        if (o.ContainsKey("write_snr_est")) snrLog.Add(rade_snrdB_3k_est(r));
        syms++;
    }
    Console.Error.WriteLine($"Processed {syms} input OFDM symbols, {valid} valid outputs");
    if (o.TryGetValue("write_snr_est", out var snrFile)) File.WriteAllBytes(snrFile!, MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(snrLog)).ToArray());
    rade_close(r);
    return 0;
}

static Stream OpenIn(string p) => p == "-" ? Console.OpenStandardInput() : File.OpenRead(p);
static Stream OpenOut(string p) => p == "-" ? Console.OpenStandardOutput() : File.Create(p);

static int LpcnetDemo(string[] a)
{
    if (a.Length != 3) return Usage();
    using var fin = OpenIn(a[1]);
    using var fout = OpenOut(a[2]);
    if (a[0] == "-features")
    {
        var enc = new LpcnetEncoder();
        var pcm = new short[LpcnetEncoder.FrameSamples];
        var f = new float[LpcnetEncoder.FeaturesPerFrame];
        while (ReadExact(fin, MemoryMarshal.AsBytes(pcm.AsSpan())) == pcm.Length * 2)
        {
            enc.ComputeFeatures(pcm, f);
            fout.Write(MemoryMarshal.AsBytes(f.AsSpan()));
        }
        return 0;
    }
    if (a[0] == "-fargan-synthesis")
    {
        var fs = new FarganStream();
        var f = new float[LpcnetConst.NbTotalFeatures];
        var pcm = new short[FarganStream.SamplesPerFrame];
        while (ReadExact(fin, MemoryMarshal.AsBytes(f.AsSpan())) == f.Length * 4)
        {
            int n = fs.Push(f, pcm);
            fout.Write(MemoryMarshal.AsBytes(pcm.AsSpan(0, n)));
        }
        return 0;
    }
    return Usage();
}

static int RadeTxWav(string[] a)
{
    var (o, pos) = Parse(a, "v", "f");
    if (pos.Count != 2) return Usage();
    int verbose = o.TryGetValue("v", out var vs) ? int.Parse(vs!, CultureInfo.InvariantCulture) : 1;
    using var br = new BinaryReader(File.OpenRead(pos[0]));
    var wav = Wav.ReadHeader(br);
    if (wav is null) { Console.Error.WriteLine($"rade_modulate: can't parse '{pos[0]}' as WAV"); return 1; }
    if (verbose >= 1)
        Console.Error.WriteLine($"Input: {pos[0]}  {wav.SampleRate} Hz  {wav.Channels} ch  {wav.BitsPerSample}-bit {(wav.IsFloat ? "float" : "int")}");
    if (wav.BitsPerSample != 16 || wav.IsFloat || wav.SampleRate != RADE_SPEECH_SAMPLE_RATE || wav.Channels != 1)
    {
        Console.Error.WriteLine("rade_modulate: input must be 16-bit PCM mono 16 kHz WAV; use sox or ffmpeg to convert");
        return 1;
    }
    var audio = Wav.ReadMonoFloat(br, wav);
    if (verbose >= 1)
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Speech input: {audio.Length} samples @ {RADE_SPEECH_SAMPLE_RATE} Hz  ({audio.Length / (double)RADE_SPEECH_SAMPLE_RATE:F1} s)"));

    var enc = new LpcnetEncoder();
    int flags = verbose < 2 ? RADE_VERBOSE_0 : 0;
    if (o.ContainsKey("v2")) flags |= RADE_MODE_V2;
    if (o.ContainsKey("no_bpf")) flags |= RADE_NO_TX_BPF;
    var r = rade_open("(unused, built-in weights)", flags)!;
    int nf = rade_n_features_in_out(r), ntx = rade_n_tx_out(r), neoo = rade_n_tx_eoo_out(r);
    int framesPerMf = nf / LpcnetConst.NbTotalFeatures;
    var featuresIn = new float[nf];
    var txOut = new RadeComp[Math.Max(ntx, neoo)];
    using var featFile = o.TryGetValue("f", out var ff) ? File.Create(ff!) : null;
    using var bw = new BinaryWriter(File.Create(pos[1]));
    Wav.WriteHeader(bw, RADE_MODEM_SAMPLE_RATE, 0);
    uint totalBytes = 0;
    void WriteIqReal(int n)
    {
        for (int i = 0; i < n; i++) bw.Write(Wav.ToInt16(txOut[i].Real * RADE_INT16_SCALE));
        totalBytes += (uint)(n * 2);
    }

    long pcmPos = 0;
    int featIdx = 0, mfCount = 0;
    var pcm = new short[LpcnetConst.LpcnetFrameSize];
    while (pcmPos + LpcnetConst.LpcnetFrameSize <= audio.Length)
    {
        for (int i = 0; i < pcm.Length; i++) pcm[i] = Wav.ToInt16(audio[pcmPos + i] * 32768.0f);
        pcmPos += LpcnetConst.LpcnetFrameSize;
        enc.ComputeFeatures(pcm, featuresIn.AsSpan(featIdx * LpcnetConst.NbTotalFeatures, LpcnetConst.NbTotalFeatures));
        if (++featIdx >= framesPerMf)
        {
            featFile?.Write(MemoryMarshal.AsBytes(featuresIn.AsSpan()));
            WriteIqReal(rade_tx(r, txOut, featuresIn));
            featIdx = 0;
            mfCount++;
        }
    }
    if (featIdx > 0)
    {
        // zero-pad remaining feature slots so the last speech segment is encoded
        Array.Clear(featuresIn, featIdx * LpcnetConst.NbTotalFeatures, (framesPerMf - featIdx) * LpcnetConst.NbTotalFeatures);
        featFile?.Write(MemoryMarshal.AsBytes(featuresIn.AsSpan(0, (framesPerMf - featIdx) * LpcnetConst.NbTotalFeatures)));
        WriteIqReal(rade_tx(r, txOut, featuresIn));
        mfCount++;
    }
    WriteIqReal(rade_tx_eoo(r, txOut));
    bw.Seek(0, SeekOrigin.Begin);
    Wav.WriteHeader(bw, RADE_MODEM_SAMPLE_RATE, totalBytes);
    if (verbose >= 1)
    {
        Console.Error.WriteLine($"Modem frames: {mfCount} + EOO");
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Output: {pos[1]}  {totalBytes / (2.0 * RADE_MODEM_SAMPLE_RATE):F1} s  ({totalBytes} bytes)"));
    }
    rade_close(r);
    return 0;
}

static int RadeRxWav(string[] a)
{
    var (o, pos) = Parse(a, "v", "f", "write_state", "write_delta_hat", "write_delta_hat_g", "write_freq_offset", "write_gain", "write_snr_est");
    if (pos.Count != 2) return Usage();
    int verbose = o.TryGetValue("v", out var vs) ? int.Parse(vs!, CultureInfo.InvariantCulture) : 1;
    using var br = new BinaryReader(File.OpenRead(pos[0]));
    var wav = Wav.ReadHeader(br);
    if (wav is null) { Console.Error.WriteLine($"rade_demod: can't parse '{pos[0]}' as WAV"); return 1; }
    if (verbose >= 1)
        Console.Error.WriteLine($"Input: {pos[0]}  {wav.SampleRate} Hz  {wav.Channels} ch  {wav.BitsPerSample}-bit {(wav.IsFloat ? "float" : "int")}");
    if (wav.BitsPerSample != 16 || wav.IsFloat || wav.SampleRate != RADE_MODEM_SAMPLE_RATE || wav.Channels != 1)
    {
        Console.Error.WriteLine("rade_demod: input must be 16-bit PCM mono 8 kHz WAV; use sox or ffmpeg to convert");
        return 1;
    }
    var audio = Wav.ReadModemReal(br, wav);
    if (verbose >= 1)
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Modem input: {audio.Length} samples @ {RADE_MODEM_SAMPLE_RATE} Hz  ({audio.Length / (double)RADE_MODEM_SAMPLE_RATE:F1} s)"));
    // real -> IQ with imag = 0: the negative-frequency image is rejected by the OFDM correlators.
    var iq = new RadeComp[audio.Length];
    for (int i = 0; i < audio.Length; i++) iq[i] = new(audio[i], 0.0f);

    int flags = 0;
    if (verbose == 0) flags |= RADE_VERBOSE_0;
    else if (verbose == 2) flags |= RADE_VERBOSE_TERSE;
    else if (verbose >= 3) flags |= RADE_VERBOSE_FULL;
    if (o.ContainsKey("v2")) flags |= RADE_MODE_V2;
    var r = rade_open("(unused, built-in weights)", flags)!;
    int ninMax = rade_nin_max(r), nf = rade_n_features_in_out(r), nbits = rade_n_eoo_bits(r);
    var rxBuf = new RadeComp[ninMax];
    var featBuf = new float[nf];
    var eooBuf = new float[nbits];

    FileStream? Diag(string k) => o.TryGetValue(k, out var p) ? File.Create(p!) : null;
    using var featFile = Diag("f");
    using var stateFp = Diag("write_state");
    using var dhFp = Diag("write_delta_hat");
    using var dhgFp = Diag("write_delta_hat_g");
    using var foFp = Diag("write_freq_offset");
    using var gainFp = Diag("write_gain");
    using var snrFp = Diag("write_snr_est");

    var fargan = new Fargan();
    bool farganReady = false;
    var contBuf = new float[5 * LpcnetConst.NbTotalFeatures];
    int contFrames = 0;
    using var bw = new BinaryWriter(File.Create(pos[1]));
    Wav.WriteHeader(bw, RADE_SPEECH_SAMPLE_RATE, 0);
    uint totalBytes = 0;
    long iqPos = 0;
    int symCount = 0, vldCount = 0;
    float snrSum = 0.0f;
    var fpcm = new float[LpcnetConst.LpcnetFrameSize];
    while (iqPos < iq.Length)
    {
        int nin = rade_nin(r);
        long remaining = iq.Length - iqPos;
        if (remaining < nin)
        {
            Array.Clear(rxBuf, 0, nin);
            Array.Copy(iq, iqPos, rxBuf, 0, remaining);
            iqPos = iq.Length;
        }
        else
        {
            Array.Copy(iq, iqPos, rxBuf, 0, nin);
            iqPos += nin;
        }
        int nOut = rade_rx(r, featBuf, out int hasEoo, eooBuf, rxBuf.AsSpan(0, nin));
        if (stateFp != null || dhFp != null || dhgFp != null || foFp != null || gainFp != null || snrFp != null)
        {
            rade_get_stats(r, out var st);
            stateFp?.Write(BitConverter.GetBytes((short)st.sync));
            dhFp?.Write(BitConverter.GetBytes(st.delta_hat));
            dhgFp?.Write(BitConverter.GetBytes(st.delta_hat_g));
            foFp?.Write(BitConverter.GetBytes(st.freq_offset));
            gainFp?.Write(BitConverter.GetBytes(st.gain));
            snrFp?.Write(BitConverter.GetBytes(st.snr_est));
        }
        if (hasEoo != 0 && verbose >= 1) Console.Error.WriteLine($"End-of-over at input OFDM symbol {symCount}");
        if (nOut > 0)
        {
            vldCount++;
            snrSum += rade_snrdB_3k_est(r);
            for (int fi = 0; fi < nOut / LpcnetConst.NbTotalFeatures; fi++)
            {
                var feat = featBuf.AsSpan(fi * LpcnetConst.NbTotalFeatures, LpcnetConst.NbTotalFeatures);
                featFile?.Write(MemoryMarshal.AsBytes(feat));
                if (!farganReady)
                {
                    feat.CopyTo(contBuf.AsSpan(contFrames * LpcnetConst.NbTotalFeatures));
                    if (++contFrames >= 5)
                    {
                        var packed = new float[5 * LpcnetConst.NbFeatures];
                        for (int i = 0; i < 5; i++)
                            contBuf.AsSpan(i * LpcnetConst.NbTotalFeatures, LpcnetConst.NbFeatures).CopyTo(packed.AsSpan(i * LpcnetConst.NbFeatures));
                        fargan.Cont(new float[Fargan.ContSamples], packed);
                        farganReady = true;
                    }
                    continue;
                }
                fargan.Synthesize(fpcm, feat);
                for (int s = 0; s < fpcm.Length; s++) bw.Write(Wav.ToInt16(fpcm[s] * 32768.0f));
                totalBytes += (uint)(fpcm.Length * 2);
            }
        }
        symCount++;
    }
    bw.Seek(0, SeekOrigin.Begin);
    Wav.WriteHeader(bw, RADE_SPEECH_SAMPLE_RATE, totalBytes);
    if (verbose >= 1)
    {
        float snrMean = vldCount != 0 ? snrSum / vldCount : 0.0f;
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Input OFDM symbols: {symCount}   valid: {vldCount}   SNR: {snrMean:F1} dB"));
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Output: {pos[1]}  {totalBytes / (2.0 * RADE_SPEECH_SAMPLE_RATE):F1} s  ({totalBytes} bytes)"));
    }
    rade_close(r);
    return 0;
}
