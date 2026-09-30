// SPDX-License-Identifier: BSD-2-Clause
// C# port of rade_c (https://github.com/freedv/rade_c), Copyright (C) 2024 David Rowe.

using System.Runtime.InteropServices;

namespace RadeSharp;

/// <summary>Single-precision complex sample, layout-compatible with the C <c>RADE_COMP</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct RadeComp(float real, float imag)
{
    public float Real = real;
    public float Imag = imag;

    public override readonly string ToString() => $"({Real}, {Imag})";
}
