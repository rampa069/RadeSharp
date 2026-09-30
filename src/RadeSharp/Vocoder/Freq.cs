// SPDX-License-Identifier: BSD-3-Clause
// Port of the analysis parts of Opus dnn/freq.c (Mozilla, Amazon):
// band energies, DCT, forward/inverse transform, LPC from cepstrum.

using RadeSharp.Nnet;
using static RadeSharp.Vocoder.LpcnetConst;

namespace RadeSharp.Vocoder;

internal sealed class Freq
{
    private static readonly short[] Eband5ms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40];
    private static readonly float[] Compensation =
        [0.8f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 0.666667f, 0.5f, 0.5f, 0.5f, 0.333333f, 0.25f, 0.25f, 0.2f, 0.166667f, 0.173913f];

    private readonly KissFft _kfft;
    private readonly float[] _halfWindow;
    private readonly float[] _dctTable;

    public Freq(WeightSet tables)
    {
        _kfft = new KissFft(tables);
        _halfWindow = tables.Find("half_window")!.AsFloats();
        _dctTable = tables.Find("dct_table")!.AsFloats();
    }

    /// <summary><c>lpcn_compute_band_energy</c>.</summary>
    public static void ComputeBandEnergy(Span<float> bandE, ReadOnlySpan<KissCpx> X)
    {
        Span<float> sum = stackalloc float[NbBands];
        sum.Clear();
        for (int i = 0; i < NbBands - 1; i++)
        {
            int bandSize = (Eband5ms[i + 1] - Eband5ms[i]) * WindowSize5ms;
            for (int j = 0; j < bandSize; j++)
            {
                float frac = (float)j / bandSize;
                var x = X[Eband5ms[i] * WindowSize5ms + j];
                float tmp = x.R * x.R;
                tmp += x.I * x.I;
                sum[i] += (1 - frac) * tmp;
                sum[i + 1] += frac * tmp;
            }
        }
        sum[0] *= 2;
        sum[NbBands - 1] *= 2;
        sum.CopyTo(bandE);
    }

    private static void InterpBandGain(Span<float> g, ReadOnlySpan<float> bandE)
    {
        // C: memset(g, 0, FREQ_SIZE) clears bytes, not floats; every used bin is written below anyway.
        for (int i = 0; i < NbBands - 1; i++)
        {
            int bandSize = (Eband5ms[i + 1] - Eband5ms[i]) * WindowSize5ms;
            for (int j = 0; j < bandSize; j++)
            {
                float frac = (float)j / bandSize;
                g[Eband5ms[i] * WindowSize5ms + j] = (1 - frac) * bandE[i] + frac * bandE[i + 1];
            }
        }
    }

    private static readonly double DctScale = Math.Sqrt(2.0 / NbBands);

    /// <summary><c>dct</c>.</summary>
    public void Dct(Span<float> output, ReadOnlySpan<float> input)
    {
        for (int i = 0; i < NbBands; i++)
        {
            float sum = 0;
            for (int j = 0; j < NbBands; j++) sum += input[j] * _dctTable[j * NbBands + i];
            output[i] = (float)(sum * DctScale);
        }
    }

    private void Idct(Span<float> output, ReadOnlySpan<float> input)
    {
        for (int i = 0; i < NbBands; i++)
        {
            float sum = 0;
            for (int j = 0; j < NbBands; j++) sum += input[j] * _dctTable[i * NbBands + j];
            output[i] = (float)(sum * DctScale);
        }
    }

    /// <summary><c>forward_transform</c>: real WINDOW_SIZE input -> FREQ_SIZE bins.</summary>
    public void ForwardTransform(Span<KissCpx> output, ReadOnlySpan<float> input)
    {
        Span<KissCpx> x = stackalloc KissCpx[WindowSize];
        Span<KissCpx> y = stackalloc KissCpx[WindowSize];
        for (int i = 0; i < WindowSize; i++) x[i] = new(input[i], 0);
        _kfft.Fft(x, y);
        y[..FreqSize].CopyTo(output);
    }

    private void InverseTransform(Span<float> output, ReadOnlySpan<KissCpx> input)
    {
        Span<KissCpx> x = stackalloc KissCpx[WindowSize];
        Span<KissCpx> y = stackalloc KissCpx[WindowSize];
        int i;
        for (i = 0; i < FreqSize; i++) x[i] = input[i];
        for (; i < WindowSize; i++) x[i] = new(x[WindowSize - i].R, -x[WindowSize - i].I);
        _kfft.Fft(x, y);
        output[0] = WindowSize * y[0].R;
        for (i = 1; i < WindowSize; i++) output[i] = WindowSize * y[WindowSize - i].R;
    }

    /// <summary><c>lpcn_lpc</c> (float build: MULT32_32_Q31 = a*b, shifts are no-ops).</summary>
    private static float Lpc(Span<float> lpc, Span<float> rc, ReadOnlySpan<float> ac, int p)
    {
        float error = ac[0];
        lpc[..p].Clear();
        rc[..p].Clear();
        if (ac[0] != 0)
        {
            for (int i = 0; i < p; i++)
            {
                float rr = 0;
                for (int j = 0; j < i; j++) rr += lpc[j] * ac[i - j];
                rr += ac[i + 1];
                float r = -rr / error;
                rc[i] = r;
                lpc[i] = r;
                for (int j = 0; j < (i + 1) >> 1; j++)
                {
                    float tmp1 = lpc[j], tmp2 = lpc[i - 1 - j];
                    lpc[j] = tmp1 + r * tmp2;
                    lpc[i - 1 - j] = tmp2 + r * tmp1;
                }
                error = error - (r * r) * error;
                if (error < .001f * ac[0]) break;
            }
        }
        return error;
    }

    private float LpcFromBands(Span<float> lpc, ReadOnlySpan<float> ex)
    {
        Span<float> ac = stackalloc float[LpcOrder + 1];
        Span<float> rc = stackalloc float[LpcOrder];
        Span<float> xr = stackalloc float[FreqSize];
        Span<KissCpx> xAuto = stackalloc KissCpx[FreqSize];
        Span<float> xauto = stackalloc float[WindowSize];
        xr.Clear();
        InterpBandGain(xr, ex);
        xr[FreqSize - 1] = 0;
        xAuto.Clear();
        for (int i = 0; i < FreqSize; i++) xAuto[i].R = xr[i];
        InverseTransform(xauto, xAuto);
        for (int i = 0; i < LpcOrder + 1; i++) ac[i] = xauto[i];
        // -40 dB noise floor.
        ac[0] = (float)(ac[0] + (ac[0] * 1e-4 + 320 / 12 / 38.0));
        // Lag windowing.
        for (int i = 1; i < LpcOrder + 1; i++) ac[i] = (float)(ac[i] * (1 - 6e-5 * i * i));
        return Lpc(lpc, rc, ac, LpcOrder);
    }

    /// <summary><c>lpc_from_cepstrum</c>. C lowers pow(10.f, x) to __exp10 (see CeltMath notes).</summary>
    public float LpcFromCepstrum(Span<float> lpc, ReadOnlySpan<float> cepstrum)
    {
        Span<float> ex = stackalloc float[NbBands];
        Span<float> tmp = stackalloc float[NbBands];
        cepstrum[..NbBands].CopyTo(tmp);
        tmp[0] += 4;
        Idct(ex, tmp);
        for (int i = 0; i < NbBands; i++) ex[i] = (float)(Math.Pow(10.0f, ex[i]) * Compensation[i]);
        return LpcFromBands(lpc, ex);
    }

    /// <summary><c>apply_window</c>.</summary>
    public void ApplyWindow(Span<float> x)
    {
        for (int i = 0; i < OverlapSize; i++)
        {
            x[i] *= _halfWindow[i];
            x[WindowSize - 1 - i] *= _halfWindow[i];
        }
    }
}
