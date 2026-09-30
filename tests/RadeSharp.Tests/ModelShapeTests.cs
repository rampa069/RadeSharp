using RadeSharp.Core;
using RadeSharp.Models;

namespace RadeSharp.Tests;

/// <summary>
/// Update guard: the hand-ported networks hard-code layer sizes (as the C code does
/// through its generated *_data.h). If regenerated weights change a shape, these
/// fail first and point at the constants in Core/RadeCoreNets.cs to update.
/// </summary>
public class ModelShapeTests
{
    [Fact]
    public void V1EncoderShapes()
    {
        var m = BuiltinWeights.RadeEnc(84);
        Assert.Equal(RadeEncoderV1.Dense1Out, m.enc_dense1.NbOutputs);
        Assert.Equal(RadeEncoderV1.ZDenseOut, m.enc_zdense.NbOutputs);
        Assert.Equal(RadeEncoderV1.GruOut, m.enc_gru1_recurrent.NbInputs);
        Assert.Equal(RadeEncoderV1.ConvOut, m.enc_conv1.NbOutputs);
        Assert.Equal(RadeEncoderV1.Dense1Out + 5 * (RadeEncoderV1.GruOut + RadeEncoderV1.ConvOut), m.enc_zdense.NbInputs);
    }

    [Fact]
    public void V1DecoderShapes()
    {
        var m = BuiltinWeights.RadeDec(84);
        Assert.Equal(RadeDecoderV1.Dense1Out, m.dec_dense1.NbOutputs);
        Assert.Equal(RadeDecoderV1.GruOut, m.dec_gru1_recurrent.NbInputs);
        Assert.Equal(RadeDecoderV1.ConvOut, m.dec_conv1.NbOutputs);
        Assert.Equal(RadeEncoderV1.ZDenseOut, m.dec_dense1.NbInputs);
        Assert.Equal(RadeDecoderV1.Dense1Out + 5 * (RadeDecoderV1.GruOut + RadeDecoderV1.ConvOut), m.dec_output.NbInputs);
    }

    [Fact]
    public void V2Shapes()
    {
        var e = BuiltinWeights.RadeEncV2.Value;
        var d = BuiltinWeights.RadeDecV2.Value;
        var s = BuiltinWeights.RadeSync.Value;
        Assert.Equal(RadeEncoderV2.Dense1Out, e.enc_v2_dense1.NbOutputs);
        Assert.Equal(RadeEncoderV2.LatentDim, e.enc_v2_zdense.NbOutputs);
        Assert.Equal(V2.RadeTxV2.LatentDimV2, RadeEncoderV2.LatentDim);
        Assert.Equal(V2.RadeTxV2.FramesPerStepV2 * V2.RadeTxV2.NumFeaturesV2, e.enc_v2_dense1.NbInputs);
        Assert.Equal(RadeDecoderV2.Dense1Out, d.dec_v2_dense1.NbOutputs);
        Assert.Equal(RadeDecoderV2.GruOut, d.dec_v2_gru1_recurrent.NbInputs);
        Assert.Equal(RadeDecoderV2.ConvOut, d.dec_v2_conv1.NbOutputs);
        Assert.Equal(V2.RadeTxV2.FramesPerStepV2 * V2.RadeTxV2.NumFeaturesV2, d.dec_v2_output.NbOutputs);
        Assert.Equal(RadeEncoderV2.LatentDim, s.sync_dense1.NbInputs);
        Assert.Equal(64, s.sync_dense1.NbOutputs);
        Assert.Equal(64, s.sync_dense2.NbOutputs);
        Assert.True(d.dec_v2_conv5.NbInputs <= Nnet.NnetOps.MaxConvInputsAll, "raise MaxConvInputsAll (rade_c opus-nnet.c.diff)");
    }

    [Fact]
    public void VocoderShapes()
    {
        var f = BuiltinWeights.Fargan.Value;
        Assert.Equal(12, f.cond_net_pembed.NbOutputs);
        Assert.Equal(320, f.cond_net_fdense2.NbOutputs);
        Assert.Equal(192, f.sig_net_fwc0_conv.NbOutputs);
        Assert.Equal(160, f.sig_net_gru1_recurrent.NbInputs);
        Assert.Equal(128, f.sig_net_gru2_recurrent.NbInputs);
        Assert.Equal(128, f.sig_net_gru3_recurrent.NbInputs);
        Assert.Equal(128, f.sig_net_skip_dense.NbOutputs);
        var p = BuiltinWeights.PitchDnn.Value;
        Assert.Equal(64, p.gru_1_recurrent.NbInputs);
        Assert.Equal(192, p.dense_final_upsampler.NbOutputs);
    }
}
