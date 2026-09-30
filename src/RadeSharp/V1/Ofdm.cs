// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_ofdm.c (Copyright (C) 2024 David Rowe): RADE V1 OFDM
// modulator/demodulator with DFT matrices and 3-pilot LS equaliser.

using static RadeSharp.Dsp.RadeDsp;

namespace RadeSharp.V1;

internal sealed class Ofdm
{
    public readonly int Bottleneck;
    public readonly float LocalPathDelayS = 0.0025f;   // 2.5 ms assumed path delay

    public readonly RadeComp[] Winv = new RadeComp[M * Nc];   // [M][Nc]: IDFT (Tx)
    public readonly RadeComp[] Wfwd = new RadeComp[Nc * M];   // [Nc][M]: DFT (Rx)
    public readonly float[] W = new float[Nc];

    public readonly RadeComp[] P = new RadeComp[Nc];
    public readonly RadeComp[] Pend = new RadeComp[Nc];
    public readonly RadeComp[] Pt = new RadeComp[M];          // p: time-domain pilot
    public readonly RadeComp[] Pendt = new RadeComp[M];       // pend
    public readonly RadeComp[] PCp = new RadeComp[M + Ncp];
    public readonly RadeComp[] PendCp = new RadeComp[M + Ncp];
    public readonly float PilotGain;

    public readonly RadeComp[] Eoo = new RadeComp[Neoo];
    public readonly int NEoo;

    public readonly RadeComp[] Pmat = new RadeComp[Nc * 2 * 3];   // [Nc][2][3]

    /// <summary><c>rade_ofdm_init</c>.</summary>
    public Ofdm(int bottleneck)
    {
        float fs = Fs;
        Bottleneck = bottleneck;

        // Centre signal on 1500 Hz; Rs' = Fs/M is the symbol rate with pilots and CP.
        float rsDash = fs / M;
        float carrier1Freq = 1500.0f - rsDash * Nc / 2.0f;
        int carrier1Index = (int)MathF.Round(carrier1Freq / rsDash, MidpointRounding.AwayFromZero);
        for (int c = 0; c < Nc; c++) W[c] = (float)(2.0f * Math.PI * (carrier1Index + c) / M);

        for (int c = 0; c < Nc; c++)
            for (int n = 0; n < M; n++)
            {
                float theta = W[c] * n;
                Winv[n * Nc + c] = Cscale(Cexp(theta), 1.0f / M);
            }
        for (int n = 0; n < M; n++)
            for (int c = 0; c < Nc; c++)
            {
                float theta = -W[c] * n;
                Wfwd[c * M + n] = Cexp(theta);
            }

        BarkerPilots(P, Nc);
        EooPilots(Pend, P, Nc);

        if (bottleneck == 3)
        {
            float pilotBackoff = MathF.Pow(10.0f, -2.0f / 20.0f);   // -2 dB backoff
            PilotGain = pilotBackoff * M / MathF.Sqrt(Nc);
        }
        else PilotGain = 1.0f;

        for (int n = 0; n < M; n++)
            for (int c = 0; c < Nc; c++)
            {
                Pt[n] = Cadd(Pt[n], Cmul(P[c], Winv[n * Nc + c]));
                Pendt[n] = Cadd(Pendt[n], Cmul(Pend[c], Winv[n * Nc + c]));
            }

        for (int n = 0; n < M; n++)
        {
            PCp[Ncp + n] = Pt[n];
            PendCp[Ncp + n] = Pendt[n];
        }
        for (int n = 0; n < Ncp; n++)
        {
            PCp[n] = Pt[M - Ncp + n];
            PendCp[n] = Pendt[M - Ncp + n];
        }

        // EOO frame: ...PE000E... -> [p_cp][pend_cp][zeros...][pend_cp]
        for (int n = 0; n < M + Ncp; n++) Eoo[n] = Cscale(PCp[n], PilotGain);
        for (int n = 0; n < M + Ncp; n++) Eoo[M + Ncp + n] = Cscale(PendCp[n], PilotGain);
        for (int n = 0; n < M + Ncp; n++) Eoo[Nmf + n] = Cscale(PendCp[n], PilotGain);
        if (bottleneck == 3)
            for (int n = 0; n < Neoo; n++) Eoo[n] = TanhLimit(Eoo[n]);
        NEoo = Nmf + M + Ncp;

        // 3-pilot LS fit h = g0 + g1*exp(-j*w[c]*a) per carrier.
        float a = LocalPathDelayS * fs;
        Span<RadeComp> A = stackalloc RadeComp[6];      // [3][2]
        Span<RadeComp> AHA = stackalloc RadeComp[4];    // [2][2]
        Span<RadeComp> AHAinv = stackalloc RadeComp[4];
        for (int c = 0; c < Nc; c++)
        {
            int cMid = c;
            if (c == 0) cMid = 1;
            if (c == Nc - 1) cMid = Nc - 2;
            for (int i = 0; i < 3; i++)
            {
                A[i * 2 + 0] = One;
                A[i * 2 + 1] = Cexp(-W[cMid - 1 + i] * a);
            }
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                {
                    AHA[i * 2 + j] = Zero;
                    for (int k = 0; k < 3; k++)
                        AHA[i * 2 + j] = Cadd(AHA[i * 2 + j], Cmul(Cconj(A[k * 2 + i]), A[k * 2 + j]));
                }
            RadeComp det = Csub(Cmul(AHA[0], AHA[3]), Cmul(AHA[1], AHA[2]));
            AHAinv[0] = Cdiv(AHA[3], det);
            AHAinv[1] = Cdiv(Cscale(AHA[1], -1.0f), det);
            AHAinv[2] = Cdiv(Cscale(AHA[2], -1.0f), det);
            AHAinv[3] = Cdiv(AHA[0], det);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                {
                    int p = (c * 2 + i) * 3 + j;
                    Pmat[p] = Zero;
                    for (int k = 0; k < 2; k++)
                        Pmat[p] = Cadd(Pmat[p], Cmul(AHAinv[i * 2 + k], Cconj(A[j * 2 + k])));
                }
        }
    }

    /// <summary><c>rade_ofdm_idft</c>: Nc carriers -> M samples.</summary>
    public void Idft(Span<RadeComp> timeOut, ReadOnlySpan<RadeComp> freqIn)
    {
        for (int n = 0; n < M; n++) timeOut[n] = CdotComp(freqIn, Winv.AsSpan(n * Nc, Nc), Nc);
    }

    /// <summary><c>rade_ofdm_insert_cp</c>.</summary>
    public static void InsertCp(Span<RadeComp> timeOut, ReadOnlySpan<RadeComp> timeIn)
    {
        timeIn.Slice(M - Ncp, Ncp).CopyTo(timeOut);
        timeIn[..M].CopyTo(timeOut[Ncp..]);
    }

    /// <summary><c>rade_ofdm_mod_frame</c>: Nzmf latent vectors -> Nmf samples.</summary>
    public int ModFrame(Span<RadeComp> txOut, ReadOnlySpan<float> z)
    {
        int outIdx = 0;
        Span<RadeComp> txSym = stackalloc RadeComp[Ns * Nc];
        int symIdx = 0;
        for (int s = 0; s < Ns; s++)
            for (int c = 0; c < Nc; c++)
            {
                int zi = symIdx * 2;
                txSym[s * Nc + c] = new(z[zi], z[zi + 1]);
                if (Bottleneck == 2) txSym[s * Nc + c] = TanhLimit(txSym[s * Nc + c]);
                symIdx++;
            }

        Span<RadeComp> pilotSym = stackalloc RadeComp[Nc];
        for (int c = 0; c < Nc; c++) pilotSym[c] = Cscale(P[c], PilotGain);

        Span<RadeComp> timeBuf = stackalloc RadeComp[M];
        Span<RadeComp> timeCp = stackalloc RadeComp[M + Ncp];
        Idft(timeBuf, pilotSym);
        InsertCp(timeCp, timeBuf);
        if (Bottleneck == 3)
            for (int n = 0; n < M + Ncp; n++) timeCp[n] = TanhLimit(timeCp[n]);
        for (int n = 0; n < M + Ncp; n++) txOut[outIdx++] = timeCp[n];

        for (int s = 0; s < Ns; s++)
        {
            Idft(timeBuf, txSym.Slice(s * Nc, Nc));
            InsertCp(timeCp, timeBuf);
            if (Bottleneck == 3)
                for (int n = 0; n < M + Ncp; n++) timeCp[n] = TanhLimit(timeCp[n]);
            for (int n = 0; n < M + Ncp; n++) txOut[outIdx++] = timeCp[n];
        }
        return Nmf;
    }

    /// <summary><c>rade_ofdm_dft</c>: M samples -> Nc carriers.</summary>
    public void Dft(Span<RadeComp> freqOut, ReadOnlySpan<RadeComp> timeIn)
    {
        for (int c = 0; c < Nc; c++) freqOut[c] = CdotComp(timeIn, Wfwd.AsSpan(c * M, M), M);
    }

    /// <summary><c>rade_ofdm_remove_cp</c>.</summary>
    public static void RemoveCp(Span<RadeComp> timeOut, ReadOnlySpan<RadeComp> timeIn, int timeOffset)
    {
        timeIn.Slice(Ncp + timeOffset, M).CopyTo(timeOut);
    }

    /// <summary><c>rade_ofdm_est_pilots</c>.</summary>
    public void EstPilots(Span<RadeComp> pilotEst, ReadOnlySpan<RadeComp> rxPilots, int numPilots)
    {
        float a = LocalPathDelayS * Fs;
        Span<RadeComp> h = stackalloc RadeComp[3];
        Span<RadeComp> g = stackalloc RadeComp[2];
        for (int p = 0; p < numPilots; p++)
        {
            var rxP = rxPilots.Slice(p * Nc, Nc);
            var estP = pilotEst.Slice(p * Nc, Nc);
            for (int c = 0; c < Nc; c++)
            {
                int cMid = c;
                if (c == 0) cMid = 1;
                if (c == Nc - 1) cMid = Nc - 2;
                for (int i = 0; i < 3; i++) h[i] = Cdiv(rxP[cMid - 1 + i], P[cMid - 1 + i]);
                for (int i = 0; i < 2; i++)
                {
                    g[i] = Zero;
                    for (int j = 0; j < 3; j++) g[i] = Cadd(g[i], Cmul(Pmat[(c * 2 + i) * 3 + j], h[j]));
                }
                estP[c] = Cadd(g[0], Cmul(g[1], Cexp(-W[c] * a)));
            }
        }
    }

    /// <summary><c>rade_ofdm_pilot_eq</c>: equalises rxSym in place, returns SNR (dB, 3 kHz).</summary>
    public float PilotEq(Span<RadeComp> rxSym, ReadOnlySpan<RadeComp> rxPilotsStart,
        ReadOnlySpan<RadeComp> pilotEstStart, ReadOnlySpan<RadeComp> pilotEstEnd, bool coarseMag)
    {
        float s1 = 0.0f, s2 = 0.0f;
        for (int c = 0; c < Nc; c++)
        {
            s1 += Cabs2(rxPilotsStart[c]);
            float rxPhase = Cangle(pilotEstStart[c]);
            RadeComp rcnHat = Cmul(rxPilotsStart[c], Cexp(-rxPhase));
            s2 += rcnHat.Imag * rcnHat.Imag;
        }
        s2 += 1e-12f;
        float snrEst = s1 / (2.0f * s2) - 1.0f;
        if (snrEst <= 0.1f) snrEst = 0.1f;
        float snrdBEst = 10.0f * MathF.Log10(snrEst);

        // Correction based on average of straight line fit to AWGN/MPG/MPP
        const float mCorr = 0.7650f, cCorr = 4.1343f;
        snrdBEst = (snrdBEst - cCorr) / mCorr;

        float rs = (float)Fs / M;
        float snrdB3k = snrdBEst + 10.0f * MathF.Log10(rs * Nc / 3000.0f) + 10.0f * MathF.Log10((float)(M + Ncp) / M);

        for (int s = 0; s < Ns; s++)
        {
            float t = (float)s / (float)(Ns + 1);
            for (int c = 0; c < Nc; c++)
            {
                RadeComp a = pilotEstStart[c], b = pilotEstEnd[c];
                RadeComp chEst = new(a.Real + t * (b.Real - a.Real), a.Imag + t * (b.Imag - a.Imag));
                float chAngle = Cangle(chEst);
                rxSym[s * Nc + c] = Cmul(rxSym[s * Nc + c], Cexp(-chAngle));
            }
        }

        if (coarseMag)
        {
            float magSum = 0.0f;
            for (int c = 0; c < Nc; c++) magSum += Cabs2(pilotEstStart[c]) + Cabs2(pilotEstEnd[c]);
            float mag = MathF.Sqrt(magSum / (2.0f * Nc)) + 1e-6f;
            if (Bottleneck == 3) mag = mag * Cabs(P[0]) / PilotGain;
            float invMag = 1.0f / mag;
            for (int s = 0; s < Ns; s++)
                for (int c = 0; c < Nc; c++) rxSym[s * Nc + c] = Cscale(rxSym[s * Nc + c], invMag);
        }
        return snrdB3k;
    }

    /// <summary><c>rade_ofdm_demod_frame</c>: returns number of latent floats written.</summary>
    public int DemodFrame(Span<float> zHat, ReadOnlySpan<RadeComp> rxIn, int timeOffset, bool endOfOver, bool coarseMag, ref float snrEst)
    {
        Span<RadeComp> rxSym = stackalloc RadeComp[(Ns + 2) * Nc];
        Span<RadeComp> timeBuf = stackalloc RadeComp[M];
        for (int s = 0; s < Ns + 2; s++)
        {
            RemoveCp(timeBuf, rxIn[(s * (M + Ncp))..], timeOffset);
            Dft(rxSym.Slice(s * Nc, Nc), timeBuf);
        }

        if (endOfOver) return DemodEoo(zHat, rxIn, timeOffset);

        Span<RadeComp> pilotEst = stackalloc RadeComp[2 * Nc];
        Span<RadeComp> rxPilots = stackalloc RadeComp[2 * Nc];
        rxSym[..Nc].CopyTo(rxPilots);
        rxSym.Slice((Ns + 1) * Nc, Nc).CopyTo(rxPilots[Nc..]);
        EstPilots(pilotEst, rxPilots, 2);

        Span<RadeComp> rxData = stackalloc RadeComp[Ns * Nc];
        rxSym.Slice(Nc, Ns * Nc).CopyTo(rxData);
        snrEst = PilotEq(rxData, rxPilots[..Nc], pilotEst[..Nc], pilotEst[Nc..], coarseMag);

        int o = 0;
        for (int i = 0; i < Ns * Nc; i++)
        {
            zHat[o++] = rxData[i].Real;
            zHat[o++] = rxData[i].Imag;
        }
        return o;
    }

    /// <summary><c>rade_ofdm_demod_eoo</c>.</summary>
    public int DemodEoo(Span<float> zHat, ReadOnlySpan<RadeComp> rxIn, int timeOffset)
    {
        Span<RadeComp> rxSym = stackalloc RadeComp[(Ns + 2) * Nc];
        Span<RadeComp> timeBuf = stackalloc RadeComp[M];
        for (int s = 0; s < Ns + 2; s++)
        {
            RemoveCp(timeBuf, rxIn[(s * (M + Ncp))..], timeOffset);
            Dft(rxSym.Slice(s * Nc, Nc), timeBuf);
        }
        for (int c = 0; c < Nc; c++)
        {
            RadeComp sum = Zero;
            sum = Cadd(sum, Cdiv(rxSym[c], P[c]));
            sum = Cadd(sum, Cdiv(rxSym[Nc + c], Pend[c]));
            sum = Cadd(sum, Cdiv(rxSym[Ns * Nc + c], Pend[c]));
            float phaseOffset = Cangle(sum);
            for (int s = 0; s < Ns + 2; s++) rxSym[s * Nc + c] = Cmul(rxSym[s * Nc + c], Cexp(-phaseOffset));
        }
        int o = 0;
        for (int s = 2; s < Ns; s++)
            for (int c = 0; c < Nc; c++)
            {
                zHat[o++] = rxSym[s * Nc + c].Real;
                zHat[o++] = rxSym[s * Nc + c].Imag;
            }
        return o;
    }
}
