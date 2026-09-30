namespace RadeSharp.Tests;

public class RadeModemTests
{
    [Theory]
    [InlineData(RadeMode.V1)]
    [InlineData(RadeMode.V2)]
    public void LoopbackDecodesAndSeesEndOfOver(RadeMode mode)
    {
        RadeLog.Writer = null;
        using var tx = new RadeModem(mode);
        using var rx = new RadeModem(mode);
        var features = Fixtures.Floats("features.f32");
        var iq = new List<RadeComp>();
        var buf = new RadeComp[Math.Max(tx.TxSamplesPerFrame, tx.TxEooSamples)];
        for (int f = 0; f + tx.FeaturesPerFrame <= features.Length; f += tx.FeaturesPerFrame)
            iq.AddRange(buf.AsSpan(0, tx.Transmit(features.AsSpan(f, tx.FeaturesPerFrame), buf)));
        iq.AddRange(buf.AsSpan(0, tx.TransmitEndOfOver(buf)));
        iq.AddRange(new RadeComp[tx.TxEooSamples]);

        var samples = iq.ToArray();
        var outFeatures = new float[rx.FeaturesPerFrame];
        int pos = 0, frames = 0;
        bool eoo = false;
        while (pos + rx.RxSamplesNeeded <= samples.Length)
        {
            int nin = rx.RxSamplesNeeded;
            var res = rx.Receive(samples.AsSpan(pos, nin), outFeatures);
            pos += nin;
            if (res.HasFeatures) frames++;
            eoo |= res.EndOfOver;
        }
        Assert.True(frames > 20, $"only {frames} frames decoded");
        Assert.True(eoo, "End-of-Over not detected");
    }

    [Fact]
    public void ApiVersionMatchesC() => Assert.Equal((2, 2), (RadeApi.rade_version(), RadeApi.rade_version_minor()));
}
