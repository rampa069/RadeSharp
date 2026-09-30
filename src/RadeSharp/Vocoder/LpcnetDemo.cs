// SPDX-License-Identifier: BSD-3-Clause
// Stream helpers equivalent to rade_c's lpcnet_demo.c (-features / -fargan-synthesis).

namespace RadeSharp.Vocoder;

/// <summary>
/// Feature-stream -> speech synthesis exactly as <c>lpcnet_demo -fargan-synthesis</c>:
/// the first 5 feature frames prime FARGAN (with silence as past speech) and produce
/// no audio; every following frame yields 160 samples at 16 kHz.
/// </summary>
public sealed class FarganStream
{
    private readonly Fargan _fargan = new();
    // lpcnet_demo reads each 36-float frame at offset i*NB_FEATURES (overlapping), keep that layout.
    private readonly float[] _prime = new float[5 * LpcnetConst.NbTotalFeatures];
    private int _primed;

    public const int SamplesPerFrame = LpcnetConst.LpcnetFrameSize;

    /// <summary>True once the 5 priming frames have been consumed.</summary>
    public bool Primed => _primed == 5;

    /// <summary>Feeds one 36-float feature frame; returns the number of samples written (0 or 160).</summary>
    public int Push(ReadOnlySpan<float> features36, Span<short> pcmOut)
    {
        if (_primed < 5)
        {
            features36[..LpcnetConst.NbTotalFeatures].CopyTo(_prime.AsSpan(_primed * LpcnetConst.NbFeatures));
            if (++_primed == 5) _fargan.Cont(new float[Fargan.ContSamples], _prime);
            return 0;
        }
        _fargan.SynthesizeInt(pcmOut, features36[..LpcnetConst.NbFeatures]);
        return SamplesPerFrame;
    }
}
