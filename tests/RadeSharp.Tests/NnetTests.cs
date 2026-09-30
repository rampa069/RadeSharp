using RadeSharp.Core;
using RadeSharp.Models;

namespace RadeSharp.Tests;

public class NnetTests
{
    private const int NIn = 84;

    [Fact]
    public void WeightBlobsParseAndBind()
    {
        Assert.NotNull(BuiltinWeights.RadeEncV2.Value);
        Assert.NotNull(BuiltinWeights.RadeDecV2.Value);
        Assert.NotNull(BuiltinWeights.RadeSync.Value);
        Assert.NotNull(BuiltinWeights.Fargan.Value);
        Assert.NotNull(BuiltinWeights.PitchDnn.Value);
        Assert.NotNull(BuiltinWeights.RadeEnc(NIn));
        Assert.NotNull(BuiltinWeights.RadeDec(NIn));
    }

    [Fact]
    public void V1EncoderDecoderMatchC()
    {
        var input = Fixtures.Floats("nn_in.f32");
        int frames = input.Length / NIn;
        var enc = new RadeEncoderV1();
        var dec = new RadeDecoderV1();
        var em = BuiltinWeights.RadeEnc(NIn);
        var dm = BuiltinWeights.RadeDec(NIn);
        var lat = new float[frames * 80];
        var outp = new float[frames * NIn];
        for (int f = 0; f < frames; f++)
        {
            enc.Run(em, lat.AsSpan(f * 80, 80), input.AsSpan(f * NIn, NIn), bottleneck: 3);
            dec.Run(dm, outp.AsSpan(f * NIn, NIn), lat.AsSpan(f * 80, 80));
        }
        Exact.Equal(Fixtures.Floats("nn_v1_lat.f32"), lat, "V1 latents");
        Exact.Equal(Fixtures.Floats("nn_v1_dec.f32"), outp, "V1 decoded features");
    }

    [Fact]
    public void V2EncoderDecoderSyncMatchC()
    {
        var input = Fixtures.Floats("nn_in.f32");
        int frames = input.Length / NIn;
        var enc = new RadeEncoderV2();
        var dec = new RadeDecoderV2();
        var lat = new float[frames * 56];
        var outp = new float[frames * 84];
        var sync = new float[frames];
        for (int f = 0; f < frames; f++)
        {
            enc.Run(BuiltinWeights.RadeEncV2.Value, lat.AsSpan(f * 56, 56), input.AsSpan(f * NIn, NIn));
            dec.Run(BuiltinWeights.RadeDecV2.Value, outp.AsSpan(f * 84, 84), lat.AsSpan(f * 56, 56));
            sync[f] = RadeFrameSync.Run(BuiltinWeights.RadeSync.Value, lat.AsSpan(f * 56, 56));
        }
        Exact.Equal(Fixtures.Floats("nn_v2_lat.f32"), lat, "V2 latents");
        Exact.Equal(Fixtures.Floats("nn_v2_dec.f32"), outp, "V2 decoded features");
        Exact.Equal(Fixtures.Floats("nn_v2_sync.f32"), sync, "V2 frame sync");
    }
}
