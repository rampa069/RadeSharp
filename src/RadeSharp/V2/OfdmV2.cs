// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_v2_ofdm.c (Copyright (C) 2025 David Rowe): RADE V2 OFDM
// (14 carriers, no pilots in normal frames, EOO pilot frame).

using static RadeSharp.Dsp.RadeDsp;

namespace RadeSharp.V2;

internal sealed class OfdmV2
{
    public const int NcV2 = 14;
    public const int MV2 = 128;
    public const int NcpV2 = 32;
    public const int NsV2 = 2;
    public const int SymLen = MV2 + NcpV2;          // 160
    public const int NmfV2 = NsV2 * SymLen;         // 320
    public const int NEooSyms = 6;
    public const int NeooV2 = NEooSyms * SymLen;    // 960
    public const int NbTotalFeaturesV2 = 36;
    public const int NumUsedFeaturesV2 = 20;

    public readonly float[] W = new float[NcV2];
    public readonly RadeComp[] Winv = new RadeComp[MV2 * NcV2];   // [M][Nc]
    public readonly RadeComp[] Wfwd = new RadeComp[NcV2 * MV2];   // [Nc][M]
    public readonly RadeComp[] PhaseCorr = new RadeComp[NcV2];
    public readonly RadeComp[] Pend = new RadeComp[NcV2];
    public readonly RadeComp[] PendTd = new RadeComp[MV2];
    public readonly RadeComp[] PendCp = new RadeComp[SymLen];
    public readonly RadeComp[] Eoo = new RadeComp[NeooV2];

    /// <summary><c>rade_v2_ofdm_init</c>.</summary>
    public OfdmV2()
    {
        float fs = Fs;
        float rsDash = fs / MV2;
        float carrier1Freq = 1500.0f - rsDash * NcV2 / 2.0f;
        int carrier1Index = (int)MathF.Round(carrier1Freq / rsDash, MidpointRounding.AwayFromZero);
        for (int c = 0; c < NcV2; c++) W[c] = (float)(2.0f * Math.PI * (carrier1Index + c) / MV2);

        for (int n = 0; n < MV2; n++)
            for (int c = 0; c < NcV2; c++) Winv[n * NcV2 + c] = Cscale(Cexp(W[c] * n), 1.0f / MV2);
        for (int c = 0; c < NcV2; c++)
            for (int n = 0; n < MV2; n++) Wfwd[c * MV2 + n] = Cexp(-W[c] * n);

        // Phase correction for correct_time_offset = -8: exp(j*8*w[c])
        for (int c = 0; c < NcV2; c++) PhaseCorr[c] = Cexp(8.0f * W[c]);

        Span<RadeComp> p = stackalloc RadeComp[NcV2];
        BarkerPilots(p, NcV2);
        EooPilots(Pend, p, NcV2);

        for (int n = 0; n < MV2; n++)
            for (int c = 0; c < NcV2; c++) PendTd[n] = Cadd(PendTd[n], Cmul(Pend[c], Winv[n * NcV2 + c]));

        PendTd.AsSpan(MV2 - NcpV2, NcpV2).CopyTo(PendCp);
        PendTd.AsSpan(0, MV2).CopyTo(PendCp.AsSpan(NcpV2));

        float pilotBackoff = MathF.Pow(10.0f, -8.0f / 20.0f);
        float pilotGain = pilotBackoff * MV2 / MathF.Sqrt(NcV2);
        for (int i = 0; i < NEooSyms; i++)
            for (int n = 0; n < SymLen; n++) Eoo[i * SymLen + n] = Cscale(PendCp[n], pilotGain);
    }

    /// <summary><c>rade_v2_ofdm_mod_frame</c>: z[LATENT_DIM] -> NMF samples.</summary>
    public int ModFrame(Span<RadeComp> txOut, ReadOnlySpan<float> z)
    {
        Span<RadeComp> timeBuf = stackalloc RadeComp[MV2];
        Span<RadeComp> freqSym = stackalloc RadeComp[NcV2];
        for (int s = 0; s < NsV2; s++)
        {
            for (int c = 0; c < NcV2; c++)
            {
                int zi = (s * NcV2 + c) * 2;
                freqSym[c] = new(z[zi], z[zi + 1]);
            }
            for (int n = 0; n < MV2; n++) timeBuf[n] = CdotComp(freqSym, Winv.AsSpan(n * NcV2, NcV2), NcV2);
            var symOut = txOut.Slice(s * SymLen, SymLen);
            timeBuf.Slice(MV2 - NcpV2, NcpV2).CopyTo(symOut);
            timeBuf.CopyTo(symOut[NcpV2..]);
        }
        return NmfV2;
    }

    /// <summary><c>rade_v2_ofdm_demod_frame</c>.</summary>
    public void DemodFrame(Span<float> zHat, ReadOnlySpan<RadeComp> rxI, int timeOffset)
    {
        for (int s = 0; s < NsV2; s++)
        {
            var rxDash = rxI[(s * SymLen + NcpV2 + timeOffset)..];
            for (int c = 0; c < NcV2; c++)
            {
                RadeComp sym = CdotComp(rxDash, Wfwd.AsSpan(c * MV2, MV2), MV2);
                sym = Cmul(sym, PhaseCorr[c]);
                int zi = (s * NcV2 + c) * 2;
                zHat[zi] = sym.Real;
                zHat[zi + 1] = sym.Imag;
            }
        }
    }

    /// <summary><c>rade_v2_ofdm_eoo_metric</c>: e_cp / e_total of the estimated channel impulse response.</summary>
    public float EooMetric(ReadOnlySpan<RadeComp> rxSymTd)
    {
        Span<RadeComp> hEst = stackalloc RadeComp[NcV2];
        for (int c = 0; c < NcV2; c++)
            hEst[c] = Cdiv(CdotComp(rxSymTd, Wfwd.AsSpan(c * MV2, MV2), MV2), Pend[c]);
        float eTotal = 1e-12f, eCp = 0.0f;
        for (int n = 0; n < MV2; n++)
        {
            RadeComp h = CdotComp(hEst, Winv.AsSpan(n * NcV2, NcV2), NcV2);
            float mag2 = h.Real * h.Real + h.Imag * h.Imag;
            eTotal += mag2;
            if (n < NcpV2 || n >= MV2 - NcpV2) eCp += mag2;
        }
        return eCp / eTotal;
    }
}
