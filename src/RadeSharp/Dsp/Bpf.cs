// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_bpf.c (Copyright (C) 2024 David Rowe): complex BPF built
// as mix-down, real lowpass FIR, mix-up.

using static RadeSharp.Dsp.RadeDsp;

namespace RadeSharp.Dsp;

internal sealed class Bpf
{
    private readonly int _ntap;
    private readonly float[] _h = new float[BpfNtap];
    private readonly RadeComp[] _mem = new RadeComp[BpfNtap];
    private RadeComp _phase;
    private readonly RadeComp _phaseInc;
    private readonly int _maxLen;

    public float Alpha { get; }

    /// <summary><c>rade_bpf_init</c>.</summary>
    public Bpf(int ntap, float fsHz, float bandwidthHz, float centreFreqHz, int maxLen)
    {
        if (ntap > BpfNtap || ntap % 2 != 1) throw new ArgumentOutOfRangeException(nameof(ntap));
        _ntap = ntap;
        Alpha = (float)(2.0f * Math.PI * centreFreqHz / fsHz);
        _maxLen = maxLen;
        float b = bandwidthHz / fsHz;
        for (int i = 0; i < ntap; i++)
        {
            int n = i - (ntap - 1) / 2;
            _h[i] = b * Sinc(n * b);
        }
        Reset();
        _phaseInc = Cexp(-Alpha);
    }

    /// <summary><c>rade_bpf_reset</c>.</summary>
    public void Reset()
    {
        Array.Clear(_mem, 0, _ntap);
        _phase = One;
    }

    /// <summary><c>rade_bpf_process</c>. <paramref name="y"/> may alias <paramref name="x"/>.</summary>
    public void Process(Span<RadeComp> y, ReadOnlySpan<RadeComp> x, int n)
    {
        if (n > _maxLen) throw new ArgumentOutOfRangeException(nameof(n));
        RadeComp phase = _phase;
        for (int i = 0; i < n; i++)
        {
            phase = Cmul(phase, _phaseInc);
            RadeComp xbb = Cmul(x[i], phase);
            Array.Copy(_mem, 0, _mem, 1, _ntap - 1);
            _mem[0] = xbb;
            RadeComp ybb = CdotFloat(_mem, _h, _ntap);
            y[i] = Cmul(ybb, Cconj(phase));
        }
        float mag = Cabs(phase);
        _phase = Cscale(phase, (float)(1.0 / mag));
    }
}
