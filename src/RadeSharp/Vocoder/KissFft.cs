// SPDX-License-Identifier: BSD-3-Clause
// Port of Opus celt/kiss_fft.c (float build): opus_fft_c / opus_fft_impl with
// the radix-2/3/4/5 butterflies. Configuration (twiddles, bitrev, factors,
// scale) comes from dnn/lpcnet_tables.c via the lpcnet_tables blob.

using RadeSharp.Nnet;

namespace RadeSharp.Vocoder;

internal struct KissCpx
{
    public float R, I;
    public KissCpx(float r, float i) { R = r; I = i; }
}

internal sealed class KissFft
{
    private const int MaxFactors = 8;
    public readonly int Nfft;
    private readonly float _scale;
    private readonly int _shift;
    private readonly int[] _factors;
    private readonly int[] _bitrev;
    private readonly KissCpx[] _twiddles;

    public KissFft(WeightSet tables)
    {
        _bitrev = tables.Find("kfft_bitrev")!.AsInts();
        Nfft = _bitrev.Length;
        _factors = tables.Find("kfft_factors")!.AsInts();
        _scale = tables.Find("kfft_scale")!.AsFloats()[0];
        _shift = -1;
        var tw = tables.Find("kfft_twiddles")!.AsFloats();
        _twiddles = new KissCpx[tw.Length / 2];
        for (int i = 0; i < _twiddles.Length; i++) _twiddles[i] = new(tw[2 * i], tw[2 * i + 1]);
    }

    /// <summary><c>opus_fft_c</c>: scaled forward FFT, fin != fout.</summary>
    public void Fft(ReadOnlySpan<KissCpx> fin, Span<KissCpx> fout)
    {
        for (int i = 0; i < Nfft; i++)
        {
            KissCpx x = fin[i];
            fout[_bitrev[i]] = new(x.R * _scale, x.I * _scale);
        }
        Impl(fout);
    }

    private void Impl(Span<KissCpx> fout)
    {
        Span<int> fstride = stackalloc int[MaxFactors];
        int shift = _shift > 0 ? _shift : 0;
        int p, m, m2, L = 0;
        fstride[0] = 1;
        do
        {
            p = _factors[2 * L];
            m = _factors[2 * L + 1];
            fstride[L + 1] = fstride[L] * p;
            L++;
        } while (m != 1);
        m = _factors[2 * L - 1];
        for (int i = L - 1; i >= 0; i--)
        {
            m2 = i != 0 ? _factors[2 * i - 1] : 1;
            switch (_factors[2 * i])
            {
                case 2: Bfly2(fout, m, fstride[i]); break;
                case 4: Bfly4(fout, fstride[i] << shift, m, fstride[i], m2); break;
                case 3: Bfly3(fout, fstride[i] << shift, m, fstride[i], m2); break;
                case 5: Bfly5(fout, fstride[i] << shift, m, fstride[i], m2); break;
            }
            m = m2;
        }
    }

    private static KissCpx Mul(KissCpx a, KissCpx b) => new(a.R * b.R - a.I * b.I, a.R * b.I + a.I * b.R);
    private static KissCpx Add(KissCpx a, KissCpx b) => new(a.R + b.R, a.I + b.I);
    private static KissCpx Sub(KissCpx a, KissCpx b) => new(a.R - b.R, a.I - b.I);

    private static void Bfly2(Span<KissCpx> fout, int m, int n)
    {
        // m == 4: radix-2 only follows a radix-4 in non-custom modes
        const float tw = 0.7071067812f;
        int f = 0;
        for (int i = 0; i < n; i++)
        {
            KissCpx t = fout[f + 4];
            fout[f + 4] = Sub(fout[f], t);
            fout[f] = Add(fout[f], t);

            t = new((fout[f + 5].R + fout[f + 5].I) * tw, (fout[f + 5].I - fout[f + 5].R) * tw);
            fout[f + 5] = Sub(fout[f + 1], t);
            fout[f + 1] = Add(fout[f + 1], t);

            t = new(fout[f + 6].I, -fout[f + 6].R);
            fout[f + 6] = Sub(fout[f + 2], t);
            fout[f + 2] = Add(fout[f + 2], t);

            t = new((fout[f + 7].I - fout[f + 7].R) * tw, -(fout[f + 7].I + fout[f + 7].R) * tw);
            fout[f + 7] = Sub(fout[f + 3], t);
            fout[f + 3] = Add(fout[f + 3], t);
            f += 8;
        }
    }

    private void Bfly4(Span<KissCpx> fout, int fstride, int m, int n, int mm)
    {
        if (m == 1)
        {
            int f = 0;
            for (int i = 0; i < n; i++)
            {
                KissCpx s0 = Sub(fout[f], fout[f + 2]);
                fout[f] = Add(fout[f], fout[f + 2]);
                KissCpx s1 = Add(fout[f + 1], fout[f + 3]);
                fout[f + 2] = Sub(fout[f], s1);
                fout[f] = Add(fout[f], s1);
                s1 = Sub(fout[f + 1], fout[f + 3]);
                fout[f + 1] = new(s0.R + s1.I, s0.I - s1.R);
                fout[f + 3] = new(s0.R - s1.I, s0.I + s1.R);
                f += 4;
            }
            return;
        }
        int m2 = 2 * m, m3 = 3 * m;
        for (int i = 0; i < n; i++)
        {
            int f = i * mm;
            int tw1 = 0, tw2 = 0, tw3 = 0;
            for (int j = 0; j < m; j++)
            {
                KissCpx s0 = Mul(fout[f + m], _twiddles[tw1]);
                KissCpx s1 = Mul(fout[f + m2], _twiddles[tw2]);
                KissCpx s2 = Mul(fout[f + m3], _twiddles[tw3]);
                KissCpx s5 = Sub(fout[f], s1);
                fout[f] = Add(fout[f], s1);
                KissCpx s3 = Add(s0, s2);
                KissCpx s4 = Sub(s0, s2);
                fout[f + m2] = Sub(fout[f], s3);
                tw1 += fstride;
                tw2 += fstride * 2;
                tw3 += fstride * 3;
                fout[f] = Add(fout[f], s3);
                fout[f + m] = new(s5.R + s4.I, s5.I - s4.R);
                fout[f + m3] = new(s5.R - s4.I, s5.I + s4.R);
                ++f;
            }
        }
    }

    private void Bfly3(Span<KissCpx> fout, int fstride, int m, int n, int mm)
    {
        int m2 = 2 * m;
        KissCpx epi3 = _twiddles[fstride * m];
        for (int i = 0; i < n; i++)
        {
            int f = i * mm;
            int tw1 = 0, tw2 = 0;
            int k = m;
            do
            {
                KissCpx s1 = Mul(fout[f + m], _twiddles[tw1]);
                KissCpx s2 = Mul(fout[f + m2], _twiddles[tw2]);
                KissCpx s3 = Add(s1, s2);
                KissCpx s0 = Sub(s1, s2);
                tw1 += fstride;
                tw2 += fstride * 2;
                fout[f + m] = new(fout[f].R - s3.R * .5f, fout[f].I - s3.I * .5f);
                s0 = new(s0.R * epi3.I, s0.I * epi3.I);
                fout[f] = Add(fout[f], s3);
                fout[f + m2] = new(fout[f + m].R + s0.I, fout[f + m].I - s0.R);
                fout[f + m] = new(fout[f + m].R - s0.I, fout[f + m].I + s0.R);
                ++f;
            } while (--k != 0);
        }
    }

    private void Bfly5(Span<KissCpx> fout, int fstride, int m, int n, int mm)
    {
        KissCpx ya = _twiddles[fstride * m];
        KissCpx yb = _twiddles[fstride * 2 * m];
        for (int i = 0; i < n; i++)
        {
            int f0 = i * mm, f1 = f0 + m, f2 = f0 + 2 * m, f3 = f0 + 3 * m, f4 = f0 + 4 * m;
            for (int u = 0; u < m; ++u)
            {
                KissCpx s0 = fout[f0];
                KissCpx s1 = Mul(fout[f1], _twiddles[u * fstride]);
                KissCpx s2 = Mul(fout[f2], _twiddles[2 * u * fstride]);
                KissCpx s3 = Mul(fout[f3], _twiddles[3 * u * fstride]);
                KissCpx s4 = Mul(fout[f4], _twiddles[4 * u * fstride]);
                KissCpx s7 = Add(s1, s4);
                KissCpx s10 = Sub(s1, s4);
                KissCpx s8 = Add(s2, s3);
                KissCpx s9 = Sub(s2, s3);

                fout[f0] = new(fout[f0].R + (s7.R + s8.R), fout[f0].I + (s7.I + s8.I));

                KissCpx s5 = new(s0.R + (s7.R * ya.R + s8.R * yb.R), s0.I + (s7.I * ya.R + s8.I * yb.R));
                KissCpx s6 = new(s10.I * ya.I + s9.I * yb.I, -(s10.R * ya.I + s9.R * yb.I));
                fout[f1] = Sub(s5, s6);
                fout[f4] = Add(s5, s6);

                KissCpx s11 = new(s0.R + (s7.R * yb.R + s8.R * ya.R), s0.I + (s7.I * yb.R + s8.I * ya.R));
                KissCpx s12 = new(s9.I * ya.I - s10.I * yb.I, s10.R * yb.I - s9.R * ya.I);
                fout[f2] = Add(s11, s12);
                fout[f3] = Sub(s11, s12);
                ++f0; ++f1; ++f2; ++f3; ++f4;
            }
        }
    }
}
