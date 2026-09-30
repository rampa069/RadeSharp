// SPDX-License-Identifier: BSD-3-Clause
// Ports from Opus celt/ (float build, FLOAT_APPROX): celt_log2, celt_fir_c,
// celt_pitch_xcorr_c, celt_inner_prod_c. The unrolled C kernels accumulate each
// output sequentially, so the plain loops below give identical sums.

namespace RadeSharp.Vocoder;

internal static class CeltMath
{
    private static readonly float[] Log2XNormCoeff =
    [
        1.000000000000000000000000000f, 8.88888895511627197265625e-01f,
        8.00000000000000000000000e-01f, 7.27272748947143554687500e-01f,
        6.66666686534881591796875e-01f, 6.15384638309478759765625e-01f,
        5.71428596973419189453125e-01f, 5.33333361148834228515625e-01f,
    ];

    private static readonly float[] Log2YNormCoeff =
    [
        0.0000000000000000000000000000f, 1.699250042438507080078125e-01f,
        3.219280838966369628906250e-01f, 4.594316184520721435546875e-01f,
        5.849624872207641601562500e-01f, 7.004396915435791015625000e-01f,
        8.073549270629882812500000e-01f, 9.068905711174011230468750e-01f,
    ];

    /// <summary><c>celt_log2</c> (FLOAT_APPROX polynomial version).</summary>
    public static float Log2(float x)
    {
        uint bits = BitConverter.SingleToUInt32Bits(x);
        int integer = (int)(bits >> 23) - 127;
        bits = unchecked((uint)((int)bits - (int)((uint)integer << 23)));
        int rangeIdx = (int)((bits >> 20) & 0x7);
        float f = BitConverter.UInt32BitsToSingle(bits) * Log2XNormCoeff[rangeIdx] - 1.0625f;
        const float A0 = 8.74628424644470214843750000e-02f, A1 = 1.357829570770263671875000000000f;
        const float A2 = -6.3897705078125000000000000e-01f, A3 = 4.01971250772476196289062500e-01f;
        const float A4 = -2.8415444493293762207031250e-01f;
        f = A0 + f * (A1 + f * (A2 + f * (A3 + f * A4)));
        return integer + f + Log2YNormCoeff[rangeIdx];
    }

    /// <summary><c>celt_log10(x)</c> = 0.3010299957f * celt_log2(x) (lpcnet_enc.c).</summary>
    public static float Log10(float x) => 0.3010299957f * Log2(x);

    /// <summary><c>celt_inner_prod_c</c>.</summary>
    public static float InnerProd(ReadOnlySpan<float> x, ReadOnlySpan<float> y, int n)
    {
        float xy = 0;
        for (int i = 0; i < n; i++) xy = xy + x[i] * y[i];
        return xy;
    }

    /// <summary><c>celt_pitch_xcorr_c</c>: xcorr[i] = sum_j x[j]*y[i+j].</summary>
    public static void PitchXcorr(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> xcorr, int len, int maxPitch)
    {
        for (int i = 0; i < maxPitch; i++) xcorr[i] = InnerProd(x, y[i..], len);
    }

    /// <summary>
    /// <c>celt_fir_c</c>: y[i] = x[i] + sum_j num[ord-1-j]*x[i+j-ord]. <paramref name="xBuf"/> holds
    /// <paramref name="ord"/> samples of history followed by the N inputs (x = xBuf + ord).
    /// </summary>
    public static void Fir(ReadOnlySpan<float> xBuf, ReadOnlySpan<float> num, Span<float> y, int n, int ord)
    {
        Span<float> rnum = stackalloc float[ord];
        for (int i = 0; i < ord; i++) rnum[i] = num[ord - i - 1];
        for (int i = 0; i < n; i++)
        {
            float sum = xBuf[ord + i];
            for (int j = 0; j < ord; j++) sum = sum + rnum[j] * xBuf[i + j];
            y[i] = sum;
        }
    }
}
