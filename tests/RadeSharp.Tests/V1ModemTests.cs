using System.Runtime.InteropServices;
using RadeSharp.V1;

namespace RadeSharp.Tests;

/// <summary>Mirror of reference/tools/rade_trace.c's per-call record.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct RxRecord
{
    public int Nin, Nout, HasEoo, Sync;
    public float FreqOffset, SnrDb, DataSymbol;
    public int StSync;
    public float StDeltaHat, StDeltaHatG, StFreqOffset, StGain, StSnrEst;
    public float EooBitsSum;
}

internal static class TracePattern
{
    // Same deterministic +/-1 pattern as rade_trace.c
    public static float Pattern(int i) => ((i * 7 + (i >> 3)) % 3) != 0 ? 1.0f : -1.0f;
}

public class V1ModemTests
{
    [Fact]
    public void TxMatchesC()
    {
        RadeLog.Writer = null;
        var features = Fixtures.Floats("features.f32");
        var tx = new RadeTxV1(bottleneck: 3, auxdata: true, bpfEn: false);
        var bits = new float[tx.NEooBits];
        for (int i = 0; i < bits.Length; i++) bits[i] = TracePattern.Pattern(i);
        tx.SetEooBits(bits);

        int nf = RadeTxV1.NFeaturesIn;
        var output = new List<RadeComp>();
        var buf = new RadeComp[Math.Max(RadeTxV1.NSamplesOut, RadeTxV1.NEooOut)];
        for (int f = 0; f + nf <= features.Length; f += nf)
        {
            int n = tx.Process(buf, features.AsSpan(f, nf));
            output.AddRange(buf.AsSpan(0, n));
        }
        int ne = tx.Eoo(buf);
        output.AddRange(buf.AsSpan(0, ne));
        output.AddRange(new RadeComp[ne]);

        Exact.Equal(MemoryMarshal.Cast<RadeComp, float>(Fixtures.Iq("v1_tx.iq")),
                    MemoryMarshal.Cast<RadeComp, float>(CollectionsMarshal.AsSpan(output)), "V1 tx IQ");
    }

    [Theory]
    [InlineData("v1_tx.iq", "v1_rx")]
    [InlineData("v1_ch.iq", "v1_ch_rx")]
    public void RxMatchesC(string iqFile, string golden)
    {
        RadeLog.Writer = null;
        var iq = Fixtures.Iq(iqFile);
        var expected = Fixtures.Records<RxRecord>(golden + ".trace");
        var rx = new RadeRxV1(bottleneck: 3, auxdata: true, bpfEn: true) { Verbose = 0 };
        var features = new float[RadeRxV1.NFeaturesOut];
        var eoo = new float[RadeRxV1.NEooBits];
        var allFeatures = new List<float>();
        int pos = 0, call = 0;
        while (true)
        {
            int nin = rx.Nin;
            if (pos + nin > iq.Length) break;
            Assert.True(call < expected.Length, "more rx calls than C");
            var e = expected[call];
            Assert.Equal(e.Nin, nin);
            int ret = rx.Process(features, eoo, iq.AsSpan(pos, nin));
            pos += nin;
            int nout = (ret & 1) != 0 ? RadeRxV1.NFeaturesOut : 0;
            Assert.True(e.Nout == nout, $"call {call}: nout {nout}, C {e.Nout}");
            Assert.True(e.HasEoo == ((ret & 2) != 0 ? 1 : 0), $"call {call}: has_eoo");
            Assert.True(e.Sync == (rx.Sync ? 1 : 0), $"call {call}: sync");
            Assert.True(BitConverter.SingleToInt32Bits(e.FreqOffset) == BitConverter.SingleToInt32Bits(rx.FreqOffset), $"call {call}: foff {rx.FreqOffset:R} C {e.FreqOffset:R}");
            Assert.True(BitConverter.SingleToInt32Bits(e.SnrDb) == BitConverter.SingleToInt32Bits(rx.SnrdB3kEst), $"call {call}: snr {rx.SnrdB3kEst:R} C {e.SnrDb:R}");
            if (nout > 0) allFeatures.AddRange(features);
            if ((ret & 2) != 0)
            {
                float sum = 0;
                for (int i = 0; i < eoo.Length; i++) sum += eoo[i] * TracePattern.Pattern(i);
                Assert.Equal(e.EooBitsSum, sum);
            }
            call++;
        }
        Assert.Equal(expected.Length, call);
        Exact.Equal(Fixtures.Floats(golden + ".f32"), CollectionsMarshal.AsSpan(allFeatures), "V1 rx features");
    }
}
