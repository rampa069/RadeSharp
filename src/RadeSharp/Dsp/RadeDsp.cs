// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_dsp.h / rade_dsp.c (Copyright (C) 2024 David Rowe).
//
// Numerics note for the whole port: C promotes to double wherever a double
// literal (0.5, M_PI, 1.0/x) or a double libm call (sqrt, exp, floor) appears.
// Those spots are written with explicit (double)/(float) casts here so results
// stay bit-identical to the C build; keep them when editing.
//
// sin/cos: when C evaluates cosf(x) and sinf(x) of the same argument in one
// function (e.g. the inlined rade_cexp), clang/gcc fuse them into sincosf,
// which on macOS rounds differently from separate calls. MathF.SinCos maps to
// sincosf, so use it exactly where the C code computes both, and MathF.Sin /
// MathF.Cos where it computes only one (check with `nm` on the C objects).

using System.Runtime.CompilerServices;

namespace RadeSharp.Dsp;

internal static class RadeDsp
{
    // System constants (rade_dsp.h)
    public const int Fs = 8000;
    public const int FsSpeech = 16000;
    public const int Nc = 30;
    public const int M = 160;
    public const int Ncp = 32;
    public const int Ns = 4;
    public const int Nzmf = 3;
    public const int Nmf = (Ns + 1) * (M + Ncp);   // 960
    public const int NsymbMf = Ns + 1;
    public const int Neoo = Nmf + M + Ncp;         // 1152
    public const int LatentDim = 80;
    public const int FramesPerStep = 4;
    public const int NumFeatures = 20;
    public const int NumFeaturesAux = 21;
    public const int NbTotalFeatures = 36;
    public const int BpfNtap = 101;
    public const float AcqFrange = 100.0f;
    public const float AcqFstep = 2.5f;
    public const int AcqNfreq = 40;
    public const float AcqPacqErr1 = 0.00001f;
    public const float AcqPacqErr2 = 0.0001f;
    public const int StateSearch = 0;
    public const int StateCandidate = 1;
    public const int StateSync = 2;
    public const float Tunsync = 3.0f;
    public const int UwErrorThresh = 7;
    public const int BarkerLen = 13;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static RadeComp Cmul(RadeComp a, RadeComp b) =>
        new(a.Real * b.Real - a.Imag * b.Imag, a.Real * b.Imag + a.Imag * b.Real);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static RadeComp Cadd(RadeComp a, RadeComp b) => new(a.Real + b.Real, a.Imag + b.Imag);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static RadeComp Csub(RadeComp a, RadeComp b) => new(a.Real - b.Real, a.Imag - b.Imag);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static RadeComp Cconj(RadeComp a) => new(a.Real, -a.Imag);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Cabs(RadeComp a) => MathF.Sqrt(a.Real * a.Real + a.Imag * a.Imag);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Cabs2(RadeComp a) => a.Real * a.Real + a.Imag * a.Imag;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Cangle(RadeComp a) => MathF.Atan2(a.Imag, a.Real);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static RadeComp Cpolar(float r, float theta)
    {
        var (sin, cos) = MathF.SinCos(theta);
        return new(r * cos, r * sin);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static RadeComp Cexp(float theta)
    {
        var (sin, cos) = MathF.SinCos(theta);
        return new(cos, sin);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static RadeComp Cscale(RadeComp a, float s) => new(a.Real * s, a.Imag * s);

    public static RadeComp Cdiv(RadeComp a, RadeComp b)
    {
        float denom = b.Real * b.Real + b.Imag * b.Imag;
        return new((a.Real * b.Real + a.Imag * b.Imag) / denom, (a.Imag * b.Real - a.Real * b.Imag) / denom);
    }

    public static readonly RadeComp Zero = new(0.0f, 0.0f);
    public static readonly RadeComp One = new(1.0f, 0.0f);

    /// <summary><c>rade_cdot_comp</c>: pairwise (recursive) sum of a[i]*b[i] -- the split order matters.</summary>
    public static RadeComp CdotComp(ReadOnlySpan<RadeComp> a, ReadOnlySpan<RadeComp> b, int n)
    {
        if (n == 1) return Cmul(a[0], b[0]);
        if (n > 1)
        {
            int mid = n / 2;
            var left = CdotComp(a, b, mid);
            var right = CdotComp(a[mid..], b[mid..], n - mid);
            return Cadd(left, right);
        }
        return Zero;
    }

    /// <summary><c>rade_cdot_float</c>: pairwise sum of a[i]*b[i] with real b.</summary>
    public static RadeComp CdotFloat(ReadOnlySpan<RadeComp> a, ReadOnlySpan<float> b, int n)
    {
        if (n == 1) return new(a[0].Real * b[0], a[0].Imag * b[0]);
        if (n > 1)
        {
            int mid = n / 2;
            var left = CdotFloat(a, b, mid);
            var right = CdotFloat(a[mid..], b[mid..], n - mid);
            return Cadd(left, right);
        }
        return Zero;
    }

    /// <summary><c>rade_cdot</c>: sum(conj(a[i]) * b[i]), sequential.</summary>
    public static RadeComp Cdot(ReadOnlySpan<RadeComp> a, ReadOnlySpan<RadeComp> b, int n)
    {
        RadeComp r = Zero;
        for (int i = 0; i < n; i++)
        {
            r.Real += a[i].Real * b[i].Real + a[i].Imag * b[i].Imag;
            r.Imag += a[i].Real * b[i].Imag - a[i].Imag * b[i].Real;
        }
        return r;
    }

    /// <summary><c>rade_cmvmul</c>: y = A x, A row-major [rows x cols].</summary>
    public static void Cmvmul(Span<RadeComp> y, ReadOnlySpan<RadeComp> A, ReadOnlySpan<RadeComp> x, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            RadeComp sum = Zero;
            var row = A.Slice(r * cols, cols);
            for (int c = 0; c < cols; c++) sum = Cadd(sum, Cmul(row[c], x[c]));
            y[r] = sum;
        }
    }

    /// <summary><c>rade_cmvmul_real</c>: y = A x with real A.</summary>
    public static void CmvmulReal(Span<RadeComp> y, ReadOnlySpan<float> A, ReadOnlySpan<RadeComp> x, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            RadeComp sum = Zero;
            var row = A.Slice(r * cols, cols);
            for (int c = 0; c < cols; c++)
            {
                sum.Real += row[c] * x[c].Real;
                sum.Imag += row[c] * x[c].Imag;
            }
            y[r] = sum;
        }
    }

    /// <summary><c>rade_tanh_limit</c>: PA saturation model tanh(|z|) exp(j angle(z)).</summary>
    public static RadeComp TanhLimit(RadeComp z)
    {
        float mag = Cabs(z);
        float angle = Cangle(z);
        return Cpolar(MathF.Tanh(mag), angle);
    }

    /// <summary><c>rade_sinc</c>.</summary>
    public static float Sinc(float x)
    {
        if (MathF.Abs(x) < 1e-10f) return 1.0f;
        float pix = (float)(Math.PI * x);
        return MathF.Sin(pix) / pix;
    }

    public static float Clampf(float x, float min, float max) => x < min ? min : x > max ? max : x;

    private static readonly float[] Barker13 = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1];

    /// <summary><c>rade_barker_pilots</c>.</summary>
    public static void BarkerPilots(Span<RadeComp> p, int nc)
    {
        float scale = MathF.Sqrt(2.0f);
        for (int i = 0; i < nc; i++) p[i] = new(scale * Barker13[i % BarkerLen], 0.0f);
    }

    /// <summary><c>rade_eoo_pilots</c>: sign flip on odd carriers.</summary>
    public static void EooPilots(Span<RadeComp> pend, ReadOnlySpan<RadeComp> p, int nc)
    {
        for (int i = 0; i < nc; i++)
        {
            pend[i] = p[i];
            if (i % 2 == 1) pend[i] = new(-pend[i].Real, -pend[i].Imag);
        }
    }
}
