// SPDX-License-Identifier: BSD-3-Clause
// Port of the scalar (NO_OPTIMIZATIONS) path of Opus dnn/vec.h (Mozilla, Amazon).
//
// Exactness: every expression keeps the C evaluation order and the C types
// (float vs. double promotion, int8 wrap-around), so results match a C build
// with -ffp-contract=off bit for bit. Don't "simplify" the loops.

using System.Runtime.CompilerServices;

namespace RadeSharp.Nnet;

internal static class Vec
{
    public const int MaxInputs = 2048;   // MAX_INPUTS

    /// <summary><c>sgemv</c> (all of sgemv16x1/sgemv8x1/generic reduce to the same per-row order).</summary>
    public static void Sgemv(Span<float> output, ReadOnlySpan<float> weights, int rows, int cols, int colStride, ReadOnlySpan<float> x)
    {
        var y = output[..rows];
        y.Clear();
        for (int j = 0; j < cols; j++)
        {
            float xj = x[j];
            var w = weights.Slice(j * colStride, rows);
            for (int i = 0; i < rows; i++) y[i] += w[i] * xj;
        }
    }

    /// <summary><c>sparse_sgemv8x4</c>.</summary>
    public static void SparseSgemv8x4(Span<float> output, ReadOnlySpan<float> w, ReadOnlySpan<int> idx, int rows, ReadOnlySpan<float> x)
    {
        output[..rows].Clear();
        int ip = 0, wp = 0;
        for (int i = 0; i < rows; i += 8)
        {
            int cols = idx[ip++];
            var y = output.Slice(i, 8);
            for (int j = 0; j < cols; j++)
            {
                int pos = idx[ip++];
                float xj0 = x[pos + 0], xj1 = x[pos + 1], xj2 = x[pos + 2], xj3 = x[pos + 3];
                var ww = w.Slice(wp, 32);
                for (int k = 0; k < 8; k++) y[k] += ww[k] * xj0;
                for (int k = 0; k < 8; k++) y[k] += ww[8 + k] * xj1;
                for (int k = 0; k < 8; k++) y[k] += ww[16 + k] * xj2;
                for (int k = 0; k < 8; k++) y[k] += ww[24 + k] * xj3;
                wp += 32;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static sbyte Quantize(float v) => unchecked((sbyte)(int)Math.Floor(.5 + 127 * v));

    /// <summary><c>sparse_cgemv8x4</c> (non-SU variant: signed int8 inputs).</summary>
    public static void SparseCgemv8x4(Span<float> output, ReadOnlySpan<sbyte> w, ReadOnlySpan<int> idx, ReadOnlySpan<float> scale, int rows, int cols, ReadOnlySpan<float> input)
    {
        Span<sbyte> x = stackalloc sbyte[cols];
        output[..rows].Clear();
        for (int i = 0; i < cols; i++) x[i] = Quantize(input[i]);
        int ip = 0, wp = 0;
        for (int i = 0; i < rows; i += 8)
        {
            int colblocks = idx[ip++];
            var y = output.Slice(i, 8);
            for (int j = 0; j < colblocks; j++)
            {
                int pos = idx[ip++];
                int xj0 = x[pos + 0], xj1 = x[pos + 1], xj2 = x[pos + 2], xj3 = x[pos + 3];
                var ww = w.Slice(wp, 32);
                for (int k = 0; k < 8; k++)
                    y[k] += ww[4 * k] * xj0 + ww[4 * k + 1] * xj1 + ww[4 * k + 2] * xj2 + ww[4 * k + 3] * xj3;
                wp += 32;
            }
        }
        for (int i = 0; i < rows; i++) output[i] *= scale[i];
    }

    /// <summary><c>cgemv8x4</c> (non-SU variant).</summary>
    public static void Cgemv8x4(Span<float> output, ReadOnlySpan<sbyte> w, ReadOnlySpan<float> scale, int rows, int cols, ReadOnlySpan<float> input)
    {
        Span<sbyte> x = stackalloc sbyte[cols];
        output[..rows].Clear();
        for (int i = 0; i < cols; i++) x[i] = Quantize(input[i]);
        int wp = 0;
        for (int i = 0; i < rows; i += 8)
        {
            var y = output.Slice(i, 8);
            for (int j = 0; j < cols; j += 4)
            {
                float xj0 = x[j + 0], xj1 = x[j + 1], xj2 = x[j + 2], xj3 = x[j + 3];
                var ww = w.Slice(wp, 32);
                for (int k = 0; k < 8; k++)
                    y[k] += ww[4 * k] * xj0 + ww[4 * k + 1] * xj1 + ww[4 * k + 2] * xj2 + ww[4 * k + 3] * xj3;
                wp += 32;
            }
        }
        for (int i = 0; i < rows; i++) output[i] *= scale[i];
    }

    /// <summary><c>lpcnet_exp2</c>: polynomial 2^x approximation.</summary>
    public static float LpcnetExp2(float x)
    {
        int integer = (int)Math.Floor(x);
        if (integer < -50) return 0;
        float frac = x - integer;
        // K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2
        float f = 0.99992522f + frac * (0.69583354f + frac * (0.22606716f + 0.078024523f * frac));
        uint i = BitConverter.SingleToUInt32Bits(f);
        i = unchecked(i + (uint)(integer << 23)) & 0x7fffffff;
        return BitConverter.UInt32BitsToSingle(i);
    }

    /// <summary><c>lpcnet_exp</c>.</summary>
    public static float LpcnetExp(float x) => LpcnetExp2(x * 1.44269504f);

    /// <summary><c>tanh_approx</c>: rational approximation, clamped to [-1, 1].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float TanhApprox(float x)
    {
        const float N0 = 952.52801514f, N1 = 96.39235687f, N2 = 0.60863042f;
        const float D0 = 952.72399902f, D1 = 413.36801147f, D2 = 11.88600922f;
        float x2 = x * x;
        float num = (N2 * x2 + N1) * x2 + N0;
        float den = (D2 * x2 + D1) * x2 + D0;
        num = num * x / den;
        // MAX32(-1.f, MIN32(1.f, num)) -- ternaries, NaN falls through like C.
        float m = 1.0f < num ? 1.0f : num;
        return -1.0f > m ? -1.0f : m;
    }

    /// <summary><c>sigmoid_approx</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SigmoidApprox(float x) => .5f + .5f * TanhApprox(.5f * x);

    public static void Softmax(Span<float> y, ReadOnlySpan<float> x, int n)
    {
        for (int i = 0; i < n; i++) y[i] = LpcnetExp(x[i]);
    }

    public static void VecTanh(Span<float> y, ReadOnlySpan<float> x, int n)
    {
        for (int i = 0; i < n; i++) y[i] = TanhApprox(x[i]);
    }

    public static void VecSigmoid(Span<float> y, ReadOnlySpan<float> x, int n)
    {
        for (int i = 0; i < n; i++) y[i] = SigmoidApprox(x[i]);
    }
}
