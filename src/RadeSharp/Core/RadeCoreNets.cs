// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_enc.c / rade_dec.c (Jan Buethe, Amazon 2022) and
// rade_enc_v2.c / rade_dec_v2.c / rade_sync.c (David Rowe 2025).

using RadeSharp.Models;
using RadeSharp.Nnet;
using static RadeSharp.Nnet.NnetOps;

namespace RadeSharp.Core;

/// <summary>Shared state layout of the DenseNet-style encoder/decoder stacks.</summary>
internal abstract class CoreNetState
{
    public bool Initialized;

    // Per-stage layer lists, built on the first Run (an instance always runs one model).
    protected LinearLayer[]? _gin, _grec, _glu, _conv;
    public readonly float[][] Gru = new float[5][];
    public readonly float[][] Conv = new float[5][];

    /// <summary><c>conv1_cond_init</c>: clears the conv memory before first use.</summary>
    protected void Conv1CondInit(float[] mem, int len, int dilation)
    {
        if (!Initialized)
            for (int i = 0; i < dilation; i++) Array.Clear(mem, i * len, len);
        Initialized = true;
    }
}

/// <summary>RADE V1 core encoder (<c>rade_core_encoder</c>).</summary>
internal sealed class RadeEncoderV1 : CoreNetState
{
    public const int Dense1Out = 64, GruOut = 64, ConvOut = 96, ZDenseOut = 80;
    private static readonly int[] ConvState = [128, 288, 448, 608, 768];
    private readonly float[] _buffer = new float[Dense1Out + 5 * GruOut + 5 * ConvOut];

    public RadeEncoderV1()
    {
        for (int i = 0; i < 5; i++)
        {
            Gru[i] = new float[GruOut];
            Conv[i] = new float[(i == 0 ? 1 : 2) * ConvState[i]];
        }
    }

    public void Run(RadeEncModel m, Span<float> latents, ReadOnlySpan<float> input, int bottleneck)
    {
        var buffer = _buffer.AsSpan();
        int o = 0;
        ComputeGenericDense(m.enc_dense1, buffer[o..], input, Activation.Tanh);
        o += Dense1Out;

        LinearLayer[] gin = _gin ??= [m.enc_gru1_input, m.enc_gru2_input, m.enc_gru3_input, m.enc_gru4_input, m.enc_gru5_input];
        LinearLayer[] grec = _grec ??= [m.enc_gru1_recurrent, m.enc_gru2_recurrent, m.enc_gru3_recurrent, m.enc_gru4_recurrent, m.enc_gru5_recurrent];
        LinearLayer[] conv = _conv ??= [m.enc_conv1, m.enc_conv2, m.enc_conv3, m.enc_conv4, m.enc_conv5];
        for (int k = 0; k < 5; k++)
        {
            ComputeGenericGru(gin[k], grec[k], Gru[k], buffer);
            Gru[k].AsSpan(0, GruOut).CopyTo(buffer[o..]);
            o += GruOut;
            if (k == 0)
            {
                Conv1CondInit(Conv[k], o, 1);
                ComputeGenericConv1d(conv[k], buffer[o..], Conv[k], buffer, o, Activation.Tanh);
            }
            else
            {
                Conv1CondInit(Conv[k], o, 2);
                ComputeGenericConv1dDilation(conv[k], buffer[o..], Conv[k], buffer, o, 2, Activation.Tanh);
            }
            o += ConvOut;
        }
        ComputeGenericDense(m.enc_zdense, latents, buffer, bottleneck == 1 ? Activation.Tanh : Activation.Linear);
    }
}

/// <summary>RADE V1 core decoder (<c>rade_core_decoder</c>).</summary>
internal sealed class RadeDecoderV1 : CoreNetState
{
    public const int Dense1Out = 96, GruOut = 96, ConvOut = 32;
    private static readonly int[] ConvState = [192, 320, 448, 576, 704];
    private readonly float[] _buffer = new float[Dense1Out + 5 * GruOut + 5 * ConvOut];

    public RadeDecoderV1()
    {
        for (int i = 0; i < 5; i++)
        {
            Gru[i] = new float[GruOut];
            Conv[i] = new float[ConvState[i]];
        }
    }

    public void Run(RadeDecModel m, Span<float> features, ReadOnlySpan<float> latents)
    {
        var buffer = _buffer.AsSpan();
        int o = 0;
        ComputeGenericDense(m.dec_dense1, buffer[o..], latents, Activation.Tanh);
        o += Dense1Out;

        LinearLayer[] gin = _gin ??= [m.dec_gru1_input, m.dec_gru2_input, m.dec_gru3_input, m.dec_gru4_input, m.dec_gru5_input];
        LinearLayer[] grec = _grec ??= [m.dec_gru1_recurrent, m.dec_gru2_recurrent, m.dec_gru3_recurrent, m.dec_gru4_recurrent, m.dec_gru5_recurrent];
        LinearLayer[] glu = _glu ??= [m.dec_glu1, m.dec_glu2, m.dec_glu3, m.dec_glu4, m.dec_glu5];
        LinearLayer[] conv = _conv ??= [m.dec_conv1, m.dec_conv2, m.dec_conv3, m.dec_conv4, m.dec_conv5];
        for (int k = 0; k < 5; k++)
        {
            ComputeGenericGru(gin[k], grec[k], Gru[k], buffer);
            ComputeGlu(glu[k], buffer[o..], Gru[k]);
            o += GruOut;
            Conv1CondInit(Conv[k], o, 1);
            ComputeGenericConv1d(conv[k], buffer[o..], Conv[k], buffer, o, Activation.Tanh);
            o += ConvOut;
        }
        ComputeGenericDense(m.dec_output, features, buffer, Activation.Linear);
    }
}

/// <summary>RADE V2 core encoder (<c>rade_core_encoder_v2</c>).</summary>
internal sealed class RadeEncoderV2 : CoreNetState
{
    public const int Dense1Out = 64, GruOut = 64, ConvOut = 96, LatentDim = 56;
    private static readonly int[] ConvState = [128, 288, 448, 608, 768];
    private readonly float[] _buffer = new float[Dense1Out + 5 * GruOut + 5 * ConvOut];

    public RadeEncoderV2()
    {
        for (int i = 0; i < 5; i++)
        {
            Gru[i] = new float[GruOut];
            Conv[i] = new float[(i == 0 ? 1 : 2) * ConvState[i]];
        }
    }

    public void Run(RadeEncV2Model m, Span<float> latents, ReadOnlySpan<float> input)
    {
        var buffer = _buffer.AsSpan();
        int o = 0;
        ComputeGenericDense(m.enc_v2_dense1, buffer[o..], input, Activation.Tanh);
        o += Dense1Out;

        LinearLayer[] gin = _gin ??= [m.enc_v2_gru1_input, m.enc_v2_gru2_input, m.enc_v2_gru3_input, m.enc_v2_gru4_input, m.enc_v2_gru5_input];
        LinearLayer[] grec = _grec ??= [m.enc_v2_gru1_recurrent, m.enc_v2_gru2_recurrent, m.enc_v2_gru3_recurrent, m.enc_v2_gru4_recurrent, m.enc_v2_gru5_recurrent];
        LinearLayer[] conv = _conv ??= [m.enc_v2_conv1, m.enc_v2_conv2, m.enc_v2_conv3, m.enc_v2_conv4, m.enc_v2_conv5];
        for (int k = 0; k < 5; k++)
        {
            ComputeGenericGru(gin[k], grec[k], Gru[k], buffer);
            Gru[k].AsSpan(0, GruOut).CopyTo(buffer[o..]);
            o += GruOut;
            if (k == 0)
            {
                Conv1CondInit(Conv[k], o, 1);
                ComputeGenericConv1d(conv[k], buffer[o..], Conv[k], buffer, o, Activation.Tanh);
            }
            else
            {
                Conv1CondInit(Conv[k], o, 2);
                ComputeGenericConv1dDilation(conv[k], buffer[o..], Conv[k], buffer, o, 2, Activation.Tanh);
            }
            o += ConvOut;
        }
        // bottleneck=0: linear activation on z_dense output (no tanh)
        ComputeGenericDense(m.enc_v2_zdense, latents, buffer, Activation.Linear);
    }
}

/// <summary>RADE V2 core decoder (<c>rade_core_decoder_v2</c>).</summary>
internal sealed class RadeDecoderV2 : CoreNetState
{
    public const int Dense1Out = 128, GruOut = 128, ConvOut = 32;
    private static readonly int[] ConvState = [256, 416, 576, 736, 896];
    private readonly float[] _buffer = new float[Dense1Out + 5 * GruOut + 5 * ConvOut];

    public RadeDecoderV2()
    {
        for (int i = 0; i < 5; i++)
        {
            Gru[i] = new float[GruOut];
            Conv[i] = new float[ConvState[i]];
        }
    }

    public void Run(RadeDecV2Model m, Span<float> features, ReadOnlySpan<float> latents)
    {
        var buffer = _buffer.AsSpan();
        int o = 0;
        ComputeGenericDense(m.dec_v2_dense1, buffer[o..], latents, Activation.Tanh);
        o += Dense1Out;

        LinearLayer[] gin = _gin ??= [m.dec_v2_gru1_input, m.dec_v2_gru2_input, m.dec_v2_gru3_input, m.dec_v2_gru4_input, m.dec_v2_gru5_input];
        LinearLayer[] grec = _grec ??= [m.dec_v2_gru1_recurrent, m.dec_v2_gru2_recurrent, m.dec_v2_gru3_recurrent, m.dec_v2_gru4_recurrent, m.dec_v2_gru5_recurrent];
        LinearLayer[] glu = _glu ??= [m.dec_v2_glu1, m.dec_v2_glu2, m.dec_v2_glu3, m.dec_v2_glu4, m.dec_v2_glu5];
        LinearLayer[] conv = _conv ??= [m.dec_v2_conv1, m.dec_v2_conv2, m.dec_v2_conv3, m.dec_v2_conv4, m.dec_v2_conv5];
        for (int k = 0; k < 5; k++)
        {
            ComputeGenericGru(gin[k], grec[k], Gru[k], buffer);
            ComputeGlu(glu[k], buffer[o..], Gru[k]);
            o += GruOut;
            // V2 decoder clears the full state size (conv1_cond_init(mem, STATE_SIZE, ...)).
            Conv1CondInit(Conv[k], ConvState[k], 1);
            ComputeGenericConv1d(conv[k], buffer[o..], Conv[k], buffer, o, Activation.Tanh);
            o += ConvOut;
        }
        ComputeGenericDense(m.dec_v2_output, features, buffer, Activation.Linear);
    }
}

/// <summary>FrameSyncNet (<c>rade_frame_sync</c>): 3 dense layers, returns a value in [0, 1].</summary>
internal static class RadeFrameSync
{
    public static float Run(RadeSyncModel m, ReadOnlySpan<float> latents)
    {
        Span<float> h1 = stackalloc float[64];
        Span<float> h2 = stackalloc float[64];
        Span<float> output = stackalloc float[1];
        ComputeGenericDense(m.sync_dense1, h1, latents, Activation.Relu);
        ComputeGenericDense(m.sync_dense2, h2, h1, Activation.Relu);
        ComputeGenericDense(m.sync_dense3, output, h2, Activation.Sigmoid);
        return output[0];
    }
}
