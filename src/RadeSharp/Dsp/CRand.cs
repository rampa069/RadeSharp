// SPDX-License-Identifier: BSD-2-Clause

namespace RadeSharp.Dsp;

/// <summary>
/// The C library <c>rand()</c> that rade_acq.c relies on, as a per-context
/// generator: Park-Miller "minimal standard" (macOS libc), seed 1. The
/// reference build links the same algorithm (reference/tools/crand.c) so the
/// golden vectors are platform independent. Unlike libc the state is not
/// process-global, so several contexts don't perturb each other.
/// </summary>
internal sealed class CRand
{
    public const int RandMax = 0x7fffffff;
    private long _next = 1;

    public int Next()
    {
        if (_next == 0) _next = 123459876;
        long hi = _next / 127773;
        long lo = _next % 127773;
        long x = 16807 * lo - 2836 * hi;
        if (x < 0) x += 0x7fffffff;
        _next = x;
        return (int)(x % ((long)RandMax + 1));
    }
}
