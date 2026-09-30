// SPDX-License-Identifier: LGPL-2.1-or-later
// Port of codec2 src/gp_interleaver.c (David Rowe): Golden Prime interleaver.

using RadeSharp;

namespace RadeSharp.Text;

internal static class GpInterleaver
{
    private static readonly int[] BTable =
    [
        56, 37,     // 700E:   HRA_56_56
        106, 67,    // 2020B:  (112,56) partial protection
        112, 71,    // 700D:   HRA_112_112
        128, 83,    // datac0: H_128_256_5
        192, 127,   // datac13: H_256_512_4, 128 data bits used
        210, 131,   // 2020:   HRAb_396_504 with 312 data bits used
        736, 457,   // datac4: H_1024_2048_4f, 448 data bits used
        1024, 641,  // datac3: H_1024_2048_4f
        1290, 797,  // datac2: H2064_516_sparse
        4096, 2531, // datac1: H_4096_8192_3d
    ];

    public static int ChooseB(int nbits)
    {
        for (int i = 0; i < BTable.Length; i += 2)
            if (BTable[i] == nbits) return BTable[i + 1];
        throw new ArgumentException($"gp_interleaver: Nbits: {nbits}, b not found!");
    }

    /// <summary><c>gp_deinterleave_comp</c>.</summary>
    public static void DeinterleaveComp(Span<RadeComp> frame, ReadOnlySpan<RadeComp> interleaved, int nbits)
    {
        int b = ChooseB(nbits);
        for (int i = 0; i < nbits; i++) frame[i] = interleaved[(b * i) % nbits];
    }

    /// <summary><c>gp_interleave_comp</c>.</summary>
    public static void InterleaveComp(Span<RadeComp> interleaved, ReadOnlySpan<RadeComp> frame, int nbits)
    {
        int b = ChooseB(nbits);
        for (int i = 0; i < nbits; i++) interleaved[(b * i) % nbits] = frame[i];
    }

    /// <summary><c>gp_interleave_bits</c>: interleaves pairs of bits (one QPSK symbol each).</summary>
    public static void InterleaveBits(Span<byte> interleaved, ReadOnlySpan<byte> frame, int nbits)
    {
        Span<byte> temp = stackalloc byte[nbits];
        int b = ChooseB(nbits);
        for (int i = 0; i < nbits; i++)
            temp[(b * i) % nbits] = (byte)(((frame[i * 2] & 1) << 1) | (frame[i * 2 + 1] & 1));
        for (int i = 0; i < nbits; i++)
        {
            interleaved[i * 2] = (byte)(temp[i] >> 1);
            interleaved[i * 2 + 1] = (byte)(temp[i] & 1);
        }
    }

    /// <summary><c>gp_deinterleave_bits</c>.</summary>
    public static void DeinterleaveBits(Span<byte> frame, ReadOnlySpan<byte> interleaved, int nbits)
    {
        Span<byte> temp = stackalloc byte[nbits];
        int b = ChooseB(nbits);
        for (int i = 0; i < nbits; i++)
        {
            int j = (b * i) % nbits;
            temp[i] = (byte)(((interleaved[j * 2] & 1) << 1) | (interleaved[j * 2 + 1] & 1));
        }
        for (int i = 0; i < nbits; i++)
        {
            frame[i * 2] = (byte)(temp[i] >> 1);
            frame[i * 2 + 1] = (byte)(temp[i] & 1);
        }
    }
}
