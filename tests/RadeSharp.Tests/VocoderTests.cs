using RadeSharp.Vocoder;

namespace RadeSharp.Tests;

public class VocoderTests
{
    [Fact]
    public void FeatureAnalysisMatchesC()
    {
        var speech = Fixtures.Shorts("speech.s16");
        var enc = new LpcnetEncoder();
        var features = new float[LpcnetEncoder.FeaturesPerFrame];
        var output = new List<float>();
        for (int i = 0; i + LpcnetEncoder.FrameSamples <= speech.Length; i += LpcnetEncoder.FrameSamples)
        {
            enc.ComputeFeatures(speech.AsSpan(i, LpcnetEncoder.FrameSamples), features);
            output.AddRange(features);
        }
        Exact.Equal(Fixtures.Floats("features.f32"), System.Runtime.InteropServices.CollectionsMarshal.AsSpan(output), "LPCNet features");
    }

    [Fact]
    public void FarganSynthesisMatchesC()
    {
        var features = Fixtures.Floats("features.f32");
        var expected = Fixtures.Shorts("fargan.s16");
        var fs = new FarganStream();
        var pcm = new short[FarganStream.SamplesPerFrame];
        var output = new List<short>();
        for (int f = 0; f + 36 <= features.Length; f += 36)
        {
            int n = fs.Push(features.AsSpan(f, 36), pcm);
            output.AddRange(pcm.AsSpan(0, n));
        }
        Assert.Equal(expected.Length, output.Count);
        int diffs = 0, first = -1;
        for (int i = 0; i < expected.Length; i++)
            if (expected[i] != output[i]) { diffs++; if (first < 0) first = i; }
        Assert.True(diffs == 0, $"{diffs} samples differ, first at {first} ({(first >= 0 ? expected[first] : 0)} vs {(first >= 0 ? output[first] : 0)})");
    }
}
