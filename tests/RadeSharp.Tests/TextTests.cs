using RadeSharp.Text;

namespace RadeSharp.Tests;

public class TextTests
{
    private const int NFloats = 180;
    private static readonly string[] Calls = ["EA5IUE", "N9WAR", "vk5dgr", "G8SEZ/P", "AB1CDEFGH", "K0PFX", "", "2E0ABC", "W1AW", "JH0VEQ"];
    private const int NNoise = 8;

    [Fact]
    public void EncoderMatchesC()
    {
        var expected = Fixtures.Floats("text_tx.f32");
        var tx = RadeText.rade_text_create();
        var syms = new float[NFloats];
        for (int c = 0; c < Calls.Length; c++)
        {
            RadeText.rade_text_generate_tx_string(tx, Calls[c], Calls[c].Length, syms, NFloats);
            Exact.Equal(expected.AsSpan(c * NFloats, NFloats), syms, $"text tx '{Calls[c]}'");
        }
    }

    [Fact]
    public void DecoderMatchesC()
    {
        var input = Fixtures.Floats("text_rx_in.f32");
        var expected = File.ReadAllLines(Path.Combine(Fixtures.Dir, "text_rx.txt"));
        var rx = RadeText.rade_text_create();
        string last = "-";
        RadeText.rade_text_set_rx_callback(rx, (_, text, _) => last = text, null);
        for (int k = 0; k < Calls.Length * NNoise; k++)
        {
            last = "-";
            RadeText.rade_text_rx(rx, input.AsSpan(k * NFloats, NFloats), NFloats / 2);
            Assert.True(expected[k] == last, $"case {k} ('{Calls[k / NNoise]}', noise #{k % NNoise}): '{last}', C '{expected[k]}'");
        }
    }

    [Fact]
    public void CallsignRoundTripsThroughRadeV1EndOfOver()
    {
        RadeLog.Writer = null;
        using var tx = new RadeModem(RadeMode.V1);
        using var rx = new RadeModem(RadeMode.V1);
        var text = new RadeText();
        var bits = new float[tx.EooBitCount];
        text.GenerateTxString("EA5IUE", 6, bits, bits.Length);
        tx.SetEooBits(bits);

        var features = Fixtures.Floats("features.f32");
        var iq = new List<RadeComp>();
        var buf = new RadeComp[Math.Max(tx.TxSamplesPerFrame, tx.TxEooSamples)];
        for (int f = 0; f + tx.FeaturesPerFrame <= features.Length; f += tx.FeaturesPerFrame)
            iq.AddRange(buf.AsSpan(0, tx.Transmit(features.AsSpan(f, tx.FeaturesPerFrame), buf)));
        iq.AddRange(buf.AsSpan(0, tx.TransmitEndOfOver(buf)));
        iq.AddRange(new RadeComp[tx.TxEooSamples]);

        string? got = null;
        var decoder = new RadeText();
        RadeText.rade_text_set_rx_callback(decoder, (_, t, _) => got = t, null);
        var samples = iq.ToArray();
        var outFeatures = new float[rx.FeaturesPerFrame];
        var eoo = new float[rx.EooBitCount];
        for (int pos = 0; pos + rx.RxSamplesNeeded <= samples.Length;)
        {
            int nin = rx.RxSamplesNeeded;
            var res = rx.Receive(samples.AsSpan(pos, nin), outFeatures, eoo);
            pos += nin;
            if (res.EndOfOver) decoder.Rx(eoo, eoo.Length / 2);
        }
        Assert.Equal("EA5IUE", got);
    }
}

public class RadeVoiceModemTests
{
    [Fact]
    public void SpeechLoopbackWithCallsign()
    {
        RadeLog.Writer = null;
        using var tx = new RadeVoiceModem();
        using var rx = new RadeVoiceModem();
        tx.SetTxCallsign("EA5IUE");
        var speech = Fixtures.Shorts("speech.s16");
        var iq = new List<RadeComp>();
        var buf = new RadeComp[Math.Max(tx.TxSamplesPerFrame, tx.TxEooSamples)];
        for (int i = 0; i + tx.SpeechSamplesPerTx <= speech.Length; i += tx.SpeechSamplesPerTx)
            iq.AddRange(buf.AsSpan(0, tx.Transmit(speech.AsSpan(i, tx.SpeechSamplesPerTx), buf)));
        iq.AddRange(buf.AsSpan(0, tx.TransmitEndOfOver(buf)));
        iq.AddRange(new RadeComp[tx.TxEooSamples]);

        var samples = iq.ToArray();
        var pcm = new short[rx.MaxPcmPerReceive];
        long pcmTotal = 0;
        string? call = null;
        for (int pos = 0; pos + rx.RxSamplesNeeded <= samples.Length;)
        {
            int nin = rx.RxSamplesNeeded;
            pcmTotal += rx.Receive(samples.AsSpan(pos, nin), pcm);
            pos += nin;
            call ??= rx.TakeEooCallsign();
        }
        Assert.True(pcmTotal > 16000 * 5, $"only {pcmTotal} samples of speech");
        Assert.Equal("EA5IUE", call);
    }
}
