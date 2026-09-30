using System.Runtime.InteropServices;
using static RadeSharp.RadeApi;

namespace RadeSharp.Tests;

/// <summary>Drives rade_api exactly like reference/tools/rade_trace.c and compares every call.</summary>
public class ApiTraceTests
{
    private static int Bits(float f) => BitConverter.SingleToInt32Bits(f);

    [Theory]
    [InlineData(false, "v1_tx.iq")]
    [InlineData(true, "v2_tx.iq")]
    public void TxMatchesC(bool v2, string golden)
    {
        RadeLog.Writer = null;
        var r = rade_open("", RADE_VERBOSE_0 | (v2 ? RADE_MODE_V2 : 0))!;
        var features = Fixtures.Floats("features.f32");
        int nf = rade_n_features_in_out(r), ntx = rade_n_tx_out(r), neoo = rade_n_tx_eoo_out(r);
        var buf = new RadeComp[Math.Max(ntx, neoo)];
        int nbits = rade_n_eoo_bits(r);
        if (nbits > 0)
        {
            var bits = new float[nbits];
            for (int i = 0; i < nbits; i++) bits[i] = TracePattern.Pattern(i);
            rade_tx_set_eoo_bits(r, bits);
        }
        var output = new List<RadeComp>();
        int frame = 0;
        for (int f = 0; f + nf <= features.Length; f += nf)
        {
            rade_tx_set_data_symbol(r, TracePattern.Pattern(frame++));
            int n = rade_tx(r, buf, features.AsSpan(f, nf));
            output.AddRange(buf.AsSpan(0, n));
        }
        int ne = rade_tx_eoo(r, buf);
        output.AddRange(buf.AsSpan(0, ne));
        output.AddRange(new RadeComp[ne]);
        rade_close(r);
        Exact.Equal(MemoryMarshal.Cast<RadeComp, float>(Fixtures.Iq(golden)),
                    MemoryMarshal.Cast<RadeComp, float>(CollectionsMarshal.AsSpan(output)), golden);
    }

    [Theory]
    [InlineData(false, "v1_tx.iq", "v1_rx")]
    [InlineData(false, "v1_ch.iq", "v1_ch_rx")]
    [InlineData(true, "v2_tx.iq", "v2_rx")]
    [InlineData(true, "v2_ch.iq", "v2_ch_rx")]
    public void RxMatchesC(bool v2, string iqFile, string golden)
    {
        RadeLog.Writer = null;
        var r = rade_open("", RADE_VERBOSE_0 | (v2 ? RADE_MODE_V2 : 0))!;
        var iq = Fixtures.Iq(iqFile);
        var expected = Fixtures.Records<RxRecord>(golden + ".trace");
        int nbits = rade_n_eoo_bits(r);
        var features = new float[rade_n_features_in_out(r)];
        var eoo = new float[nbits];
        var allFeatures = new List<float>();
        int pos = 0, call = 0;
        while (true)
        {
            int nin = rade_nin(r);
            if (pos + nin > iq.Length) break;
            Assert.True(call < expected.Length, "more rx calls than C");
            var e = expected[call];
            Assert.True(e.Nin == nin, $"call {call}: nin {nin}, C {e.Nin}");
            int nout = rade_rx(r, features, out int hasEoo, eoo, iq.AsSpan(pos, nin));
            pos += nin;
            rade_get_stats(r, out var st);
            string at = $"call {call}";
            Assert.True(e.Nout == nout, $"{at}: nout {nout}, C {e.Nout}");
            Assert.True(e.HasEoo == hasEoo, $"{at}: has_eoo {hasEoo}, C {e.HasEoo}");
            Assert.True(e.Sync == rade_sync(r), $"{at}: sync");
            Assert.True(Bits(e.FreqOffset) == Bits(rade_freq_offset(r)), $"{at}: foff {rade_freq_offset(r):R}, C {e.FreqOffset:R}");
            Assert.True(Bits(e.SnrDb) == Bits(rade_snrdB_3k_est(r)), $"{at}: snr {rade_snrdB_3k_est(r):R}, C {e.SnrDb:R}");
            Assert.True(Bits(e.DataSymbol) == Bits(rade_rx_get_data_symbol(r)), $"{at}: data symbol");
            Assert.True(e.StSync == st.sync && Bits(e.StDeltaHat) == Bits(st.delta_hat) && Bits(e.StDeltaHatG) == Bits(st.delta_hat_g)
                        && Bits(e.StFreqOffset) == Bits(st.freq_offset) && Bits(e.StGain) == Bits(st.gain) && Bits(e.StSnrEst) == Bits(st.snr_est),
                        $"{at}: stats");
            if (nout > 0) allFeatures.AddRange(features.AsSpan(0, nout));
            if (hasEoo != 0 && nbits > 0)
            {
                float sum = 0;
                for (int i = 0; i < nbits; i++) sum += eoo[i] * TracePattern.Pattern(i);
                Assert.True(Bits(e.EooBitsSum) == Bits(sum), $"{at}: eoo bits");
            }
            call++;
        }
        rade_close(r);
        Assert.Equal(expected.Length, call);
        Exact.Equal(Fixtures.Floats(golden + ".f32"), CollectionsMarshal.AsSpan(allFeatures), golden + " features");
    }
}
