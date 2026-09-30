// SPDX-License-Identifier: BSD-2-Clause
// Managed equivalent of Zeus' native zeus_rade shim (native/radae/shim/zeus_rade.c):
// RADE + FARGAN vocoder + LPCNet analyzer + FreeDV reliable-text EOO callsign
// behind one speech <-> modem-IQ streaming surface. Behaviour (priming, int16
// conversion, callsign hand-off) follows the shim so Zeus can swap it in 1:1.

using RadeSharp.Vocoder;
using static RadeSharp.RadeApi;

namespace RadeSharp.Text;

/// <summary>
/// Speech (16 kHz int16) &lt;-&gt; RADE modem IQ (8 kHz complex). RX: IQ -> features ->
/// FARGAN (first 5 frames prime the vocoder, then 160 samples per frame). TX: speech ->
/// LPCNet features -> rade_tx; the SSB audio to transmit is the real part of the IQ.
/// Not thread-safe.
/// </summary>
public sealed class RadeVoiceModem : IDisposable
{
    public const int MaxCallsignLength = RadeText.MaxLength;
    private const int PrimeFrames = 5;
    private const int MaxFramesPerRx = 64;
    private const int FrameSamples = LpcnetConst.LpcnetFrameSize;
    private const int TotalFeatures = LpcnetConst.NbTotalFeatures;

    private readonly Rade _r;
    private readonly int _nFeatures, _nEooBits;

    private readonly Fargan _fargan = new();
    private bool _primed;
    private int _primeHave;
    private readonly float[] _primeBuf = new float[PrimeFrames * TotalFeatures];
    private readonly float[] _featScratch = new float[MaxFramesPerRx * TotalFeatures];
    private readonly float[] _eooScratch = new float[1024];
    private readonly float[] _fpcm = new float[FrameSamples];
    private readonly RadeText _txtRx = new();
    private string? _callsign;

    private readonly LpcnetEncoder _enc = new();
    private readonly float[] _txFeat = new float[MaxFramesPerRx * TotalFeatures];
    private readonly RadeText _txtTx = new();
    private readonly float[] _eooTxBits = new float[1024];

    public RadeVoiceModem(RadeMode mode = RadeMode.V1)
    {
        int flags = RADE_USE_C_ENCODER | RADE_USE_C_DECODER | RADE_VERBOSE_0;
        if (mode == RadeMode.V2) flags |= RADE_MODE_V2;
        _r = rade_open("", flags) ?? throw new InvalidOperationException("rade_open failed");
        Mode = mode;
        _nFeatures = rade_n_features_in_out(_r);
        _nEooBits = rade_n_eoo_bits(_r);
        SpeechSamplesPerTx = _nFeatures / TotalFeatures * FrameSamples;
        TxSamplesPerFrame = rade_n_tx_out(_r);
        TxEooSamples = rade_n_tx_eoo_out(_r);
        RadeText.rade_text_set_rx_callback(_txtRx, (_, text, _) =>
        {
            if (text.Length > 0) _callsign = text.Length <= MaxCallsignLength ? text : text[..MaxCallsignLength];
        }, null);
    }

    public RadeMode Mode { get; }
    public Rade Handle => _r;

    /// <summary>IQ samples the next <see cref="Receive"/> consumes (varies with timing tracking).</summary>
    public int RxSamplesNeeded => rade_nin(_r);
    public int RxMaxSamples => rade_nin_max(_r);
    /// <summary>Upper bound of PCM samples produced by one <see cref="Receive"/>.</summary>
    public int MaxPcmPerReceive => MaxFramesPerRx * FrameSamples;
    /// <summary>int16 speech samples consumed per <see cref="Transmit"/>.</summary>
    public int SpeechSamplesPerTx { get; }
    public int TxSamplesPerFrame { get; }
    public int TxEooSamples { get; }

    public bool InSync => rade_sync(_r) != 0;
    public float FrequencyOffsetHz => rade_freq_offset(_r);
    /// <summary>SNR (dB, 3 kHz) truncated to int like the shim's zeus_rade_snr_db.</summary>
    public int SnrDb => (int)rade_snrdB_3k_est(_r);

    /// <summary>Decodes <see cref="RxSamplesNeeded"/> IQ samples; returns PCM samples written (0 while unsynced/priming).</summary>
    public int Receive(ReadOnlySpan<RadeComp> iqIn, Span<short> pcmOut)
    {
        int nfloats = rade_rx(_r, _featScratch, out int hasEoo, _nEooBits > 0 ? _eooScratch : default, iqIn[..RxSamplesNeeded]);

        // On loss of sync, drop the priming so we re-prime cleanly on re-acquire.
        if (rade_sync(_r) == 0)
        {
            _primed = false;
            _primeHave = 0;
        }
        if (hasEoo != 0 && _nEooBits > 0) _txtRx.Rx(_eooScratch, _nEooBits);
        if (nfloats <= 0) return 0;

        int nPcm = 0;
        int frames = Math.Min(nfloats / TotalFeatures, MaxFramesPerRx);
        for (int f = 0; f < frames; f++)
        {
            var frame36 = _featScratch.AsSpan(f * TotalFeatures, TotalFeatures);
            if (!_primed)
            {
                frame36.CopyTo(_primeBuf.AsSpan(_primeHave * LpcnetConst.NbFeatures));
                if (++_primeHave == PrimeFrames)
                {
                    // Like the shim, re-priming after a sync loss continues the same FARGAN state
                    // (fargan_init runs once, at open).
                    _fargan.Cont(new float[2 * FrameSamples], _primeBuf);
                    _primed = true;
                }
                continue;   // priming frames produce no audio
            }
            _fargan.Synthesize(_fpcm, frame36[..LpcnetConst.NbFeatures]);
            for (int i = 0; i < FrameSamples; i++)
            {
                float s = 32768.0f * _fpcm[i];
                if (s > 32767.0f) s = 32767.0f;
                if (s < -32767.0f) s = -32767.0f;
                pcmOut[nPcm++] = (short)MathF.Floor(0.5f + s);
            }
        }
        return nPcm;
    }

    /// <summary>Encodes <see cref="SpeechSamplesPerTx"/> speech samples; returns IQ samples written.</summary>
    public int Transmit(ReadOnlySpan<short> pcmIn, Span<RadeComp> txOut)
    {
        int frames = Math.Min(_nFeatures / TotalFeatures, MaxFramesPerRx);
        for (int f = 0; f < frames; f++)
            _enc.ComputeFeatures(pcmIn.Slice(f * FrameSamples, FrameSamples), _txFeat.AsSpan(f * TotalFeatures, TotalFeatures));
        return rade_tx(_r, txOut, _txFeat);
    }

    /// <summary>End-of-Over frame (carries the callsign on V1); call once on un-key.</summary>
    public int TransmitEndOfOver(Span<RadeComp> txOut) => rade_tx_eoo(_r, txOut);

    /// <summary>Callsign for the V1 EOO frame (FreeDV reliable-text, &lt;= 8 chars); null/empty clears.</summary>
    public void SetTxCallsign(string? callsign)
    {
        int n = Math.Min(_nEooBits, _eooTxBits.Length);
        Array.Clear(_eooTxBits, 0, n);
        if (!string.IsNullOrEmpty(callsign)) _txtTx.GenerateTxString(callsign, callsign.Length, _eooTxBits, n);
        rade_tx_set_eoo_bits(_r, _eooTxBits);
    }

    /// <summary>Last CRC-valid EOO callsign since the previous call, or null. Consumes it.</summary>
    public string? TakeEooCallsign()
    {
        var c = _callsign;
        _callsign = null;
        return c;
    }

    public void Dispose() => rade_close(_r);
}
