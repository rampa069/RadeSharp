// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_acq.c (Copyright (C) 2024 David Rowe): RADE V1 pilot
// acquisition (coarse time/frequency search, refinement, per-frame check).

using RadeSharp.Dsp;
using static RadeSharp.Dsp.RadeDsp;

namespace RadeSharp.V1;

internal sealed class Acq
{
    private readonly float[] _fcoarseRange = new float[AcqNfreq];
    private readonly int _nFcoarse;
    private readonly RadeComp[] _pW = new RadeComp[AcqNfreq * M];   // [nfreq][M]
    private readonly RadeComp[] _p = new RadeComp[M];
    private readonly RadeComp[] _pend = new RadeComp[M];
    private readonly RadeComp[] _dt1 = new RadeComp[Nmf * AcqNfreq];   // [Nmf][nfreq]
    private readonly RadeComp[] _dt2 = new RadeComp[Nmf * AcqNfreq];
    private readonly CRand _rand = new();

    public readonly float SigmaP;
    public float Dthresh;
    public float Dtmax12;
    public float Dtmax12Eoo;
    public int FIndMax;
    public readonly float PacqError1 = AcqPacqErr1;
    public readonly float PacqError2 = AcqPacqErr2;

    /// <summary><c>rade_acq_init</c>.</summary>
    public Acq(Ofdm ofdm, float frange, float fstep)
    {
        ofdm.Pt.CopyTo(_p, 0);
        ofdm.Pendt.CopyTo(_pend, 0);
        SigmaP = MathF.Sqrt(Cdot(ofdm.Pt, ofdm.Pt, M).Real);

        _nFcoarse = 0;
        for (float f = -frange / 2.0f; f < frange / 2.0f && _nFcoarse < AcqNfreq; f += fstep)
            _fcoarseRange[_nFcoarse++] = f;

        for (int fi = 0; fi < _nFcoarse; fi++)
        {
            float f = _fcoarseRange[fi];
            float w = (float)(2.0f * Math.PI * f / Fs);
            for (int n = 0; n < M; n++) _pW[fi * M + n] = Cmul(Cexp(w * n), _p[n]);
        }
    }

    private static readonly float SqrtPiOver2 = MathF.Sqrt((float)(Math.PI / 2.0f));

    private float SigmaR()
    {
        float sum1 = 0.0f, sum2 = 0.0f;
        int count = 0;
        for (int t = 0; t < Nmf; t++)
            for (int fi = 0; fi < _nFcoarse; fi++)
            {
                sum1 += Cabs(_dt1[t * AcqNfreq + fi]);
                sum2 += Cabs(_dt2[t * AcqNfreq + fi]);
                count++;
            }
        float sigmaR1 = (sum1 / count) / SqrtPiOver2;
        float sigmaR2 = (sum2 / count) / SqrtPiOver2;
        return (sigmaR1 + sigmaR2) / 2.0f;
    }

    private void Correlate(ReadOnlySpan<RadeComp> rx, int t, int fi, out RadeComp dt1, out RadeComp dt2)
    {
        dt1 = Zero;
        dt2 = Zero;
        var pw = _pW.AsSpan(fi * M, M);
        for (int n = 0; n < M; n++)
        {
            dt1 = Cadd(dt1, Cmul(Cconj(rx[t + n]), pw[n]));
            dt2 = Cadd(dt2, Cmul(Cconj(rx[t + Nmf + n]), pw[n]));
        }
    }

    /// <summary><c>rade_acq_detect_pilots</c>: needs 2*Nmf + M + Ncp samples.</summary>
    public bool DetectPilots(ReadOnlySpan<RadeComp> rx, out int tmax, out float fmax)
    {
        float dtmax12 = 0.0f;
        int fIndMax = 0, tMax = 0;
        float fMax = 0.0f;
        Array.Clear(_dt1);
        Array.Clear(_dt2);
        for (int t = 0; t < Nmf; t++)
            for (int fi = 0; fi < _nFcoarse; fi++)
            {
                Correlate(rx, t, fi, out var dt1, out var dt2);
                _dt1[t * AcqNfreq + fi] = dt1;
                _dt2[t * AcqNfreq + fi] = dt2;
                float dt12 = Cabs(dt1) + Cabs(dt2);
                if (dt12 > dtmax12)
                {
                    dtmax12 = dt12;
                    fIndMax = fi;
                    fMax = _fcoarseRange[fi];
                    tMax = t;
                }
            }
        float sigmaR = SigmaR();
        Dthresh = 2.0f * sigmaR * MathF.Sqrt(-MathF.Log(PacqError1 / 5.0f));
        Dtmax12 = dtmax12;
        FIndMax = fIndMax;
        tmax = tMax;
        fmax = fMax;
        return dtmax12 > Dthresh;
    }

    /// <summary><c>rade_acq_refine</c>: fine time/frequency search around the coarse estimate.</summary>
    public void Refine(ReadOnlySpan<RadeComp> rx, ref int tmax, ref float fmax,
        int tfineStart, int tfineEnd, float ffineStart, float ffineEnd, float ffineStep)
    {
        float dtmax = 0.0f;
        int tBest = tmax;
        float fBest = fmax;
        Span<RadeComp> w1p = stackalloc RadeComp[M];
        Span<RadeComp> w2p = stackalloc RadeComp[M];
        for (float f = ffineStart; f < ffineEnd; f += ffineStep)
        {
            float w = (float)(2.0f * Math.PI * f / Fs);
            for (int n = 0; n < M; n++)
            {
                RadeComp wv1 = Cexp(-w * n);
                w1p[n] = Cmul(wv1, Cconj(_p[n]));
                RadeComp wv2 = Cmul(wv1, Cexp(-w * Nmf));
                w2p[n] = Cmul(wv2, Cconj(_p[n]));
            }
            for (int t = tfineStart; t < tfineEnd; t++)
            {
                RadeComp dt1 = Zero, dt2 = Zero;
                for (int n = 0; n < M; n++)
                {
                    dt1 = Cadd(dt1, Cmul(rx[t + n], w1p[n]));
                    dt2 = Cadd(dt2, Cmul(rx[t + Nmf + n], w2p[n]));
                }
                float dt = Cabs(Cadd(dt1, dt2));
                if (dt > dtmax)
                {
                    dtmax = dt;
                    tBest = t;
                    fBest = f;
                }
            }
        }
        tmax = tBest;
        fmax = fBest;
    }

    /// <summary><c>rade_acq_check_pilots</c>.</summary>
    public bool CheckPilots(ReadOnlySpan<RadeComp> rx, int tmax, float fmax, out bool valid, out bool endOfOver)
    {
        float fs = Fs;
        // Update 5% of the correlation grid for noise estimation
        int nUpdate = (int)(0.05f * Nmf);
        for (int i = 0; i < nUpdate; i++)
        {
            int t = _rand.Next() % Nmf;
            for (int fi = 0; fi < _nFcoarse; fi++)
            {
                Correlate(rx, t, fi, out var dt1, out var dt2);
                _dt1[t * AcqNfreq + fi] = dt1;
                _dt2[t * AcqNfreq + fi] = dt2;
            }
        }

        float sigmaR = SigmaR();
        Dthresh = 2.0f * sigmaR * MathF.Sqrt(-MathF.Log(PacqError2 / 5.0f));
        float dthreshEoo = 2.0f * sigmaR * MathF.Sqrt(-MathF.Log(PacqError1 / 5.0f));

        float w = (float)(2.0f * Math.PI * fmax / fs);
        Span<RadeComp> wv = stackalloc RadeComp[M];
        for (int n = 0; n < M; n++) wv[n] = Cexp(-w * n);

        RadeComp d1 = Zero, d2 = Zero;
        for (int n = 0; n < M; n++)
        {
            RadeComp rs = Cmul(wv[n], rx[tmax + n]);
            d1 = Cadd(d1, Cmul(Cconj(rs), _p[n]));
            rs = Cmul(wv[n], rx[tmax + Nmf + n]);
            d2 = Cadd(d2, Cmul(Cconj(rs), _p[n]));
        }
        float dtmax12 = Cabs(d1) + Cabs(d2);
        Dtmax12 = dtmax12;

        RadeComp e1 = Zero, e2 = Zero;
        for (int n = 0; n < M; n++)
        {
            RadeComp rs = Cmul(wv[n], rx[tmax + M + Ncp + n]);
            e1 = Cadd(e1, Cmul(Cconj(rs), _pend[n]));
            rs = Cmul(wv[n], rx[tmax + Nmf + n]);
            e2 = Cadd(e2, Cmul(Cconj(rs), _pend[n]));
        }
        float dtmax12Eoo = Cabs(e1) + Cabs(e2);
        Dtmax12Eoo = dtmax12Eoo;

        valid = dtmax12 > Dthresh;
        endOfOver = dtmax12Eoo > dthreshEoo;
        return valid;
    }
}
