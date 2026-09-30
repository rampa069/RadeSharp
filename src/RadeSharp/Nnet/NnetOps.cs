// SPDX-License-Identifier: BSD-3-Clause
// Port of Opus dnn/nnet.c + dnn/nnet_arch.h (Mozilla, Amazon), scalar path,
// including rade_c's opus-nnet.c.diff (MAX_CONV_INPUTS_ALL raised to 2048).

namespace RadeSharp.Nnet;

/// <summary>Activation functions (<c>ACTIVATION_*</c>).</summary>
public enum Activation
{
    Linear = 0,
    Sigmoid = 1,
    Tanh = 2,
    Relu = 3,
    Softmax = 4,
    Swish = 5,
    Exp = 6,
}

/// <summary>The Opus DNN primitives RADE, FARGAN and PitchDNN are built from.</summary>
public static class NnetOps
{
    public const int MaxConvInputsAll = 2048;   // MAX_CONV_INPUTS_ALL (patched by rade_c)
    private const int MaxActivations = 4096;
    private const int MaxConv2dInputs = 8192;

    /// <summary><c>compute_activation</c>. <paramref name="output"/> may alias <paramref name="input"/>.</summary>
    public static void ComputeActivation(Span<float> output, ReadOnlySpan<float> input, int n, Activation activation)
    {
        switch (activation)
        {
            case Activation.Sigmoid:
                Vec.VecSigmoid(output, input, n);
                break;
            case Activation.Tanh:
                Vec.VecTanh(output, input, n);
                break;
            case Activation.Swish:
            {
                if (n > MaxActivations) throw new ArgumentOutOfRangeException(nameof(n));
                Span<float> tmp = stackalloc float[n];
                Vec.VecSigmoid(tmp, input, n);
                for (int i = 0; i < n; i++) output[i] = input[i] * tmp[i];
                break;
            }
            case Activation.Relu:
                for (int i = 0; i < n; i++) output[i] = input[i] < 0 ? 0 : input[i];
                break;
            case Activation.Softmax:
            {
                float sum = 0;
                Vec.Softmax(output, input, n);
                for (int i = 0; i < n; i++) sum += output[i];
                sum = (float)(1.0f / (sum + 1e-30));
                for (int i = 0; i < n; i++) output[i] = sum * output[i];
                break;
            }
            case Activation.Exp:
                Vec.Softmax(output, input, n);
                break;
            default:
                if (activation != Activation.Linear) throw new ArgumentOutOfRangeException(nameof(activation));
                input[..n].CopyTo(output);
                break;
        }
    }

    /// <summary><c>compute_linear</c>. <paramref name="output"/> must not alias <paramref name="input"/>.</summary>
    public static void ComputeLinear(LinearLayer linear, Span<float> output, ReadOnlySpan<float> input)
    {
        float[]? bias = linear.Bias;
        int m = linear.NbInputs;
        int n = linear.NbOutputs;
        if (linear.FloatWeights != null)
        {
            if (linear.WeightsIdx != null) Vec.SparseSgemv8x4(output, linear.FloatWeights, linear.WeightsIdx, n, input);
            else Vec.Sgemv(output, linear.FloatWeights, n, m, n, input);
        }
        else if (linear.Weights != null)
        {
            if (linear.WeightsIdx != null) Vec.SparseCgemv8x4(output, linear.Weights, linear.WeightsIdx, linear.Scale, n, m, input);
            else Vec.Cgemv8x4(output, linear.Weights, linear.Scale, n, m, input);
            // USE_SU_BIAS is only defined on x86 SIMD builds; the scalar path keeps the plain bias.
        }
        else output[..n].Clear();
        if (bias != null)
        {
            for (int i = 0; i < n; i++) output[i] += bias[i];
        }
        if (linear.Diag != null)
        {
            // Diag is only used for GRU recurrent weights.
            var diag = linear.Diag;
            for (int i = 0; i < m; i++)
            {
                output[i] += diag[i] * input[i];
                output[i + m] += diag[i + m] * input[i];
                output[i + 2 * m] += diag[i + 2 * m] * input[i];
            }
        }
    }

    /// <summary><c>compute_generic_dense</c>.</summary>
    public static void ComputeGenericDense(LinearLayer layer, Span<float> output, ReadOnlySpan<float> input, Activation activation)
    {
        ComputeLinear(layer, output, input);
        ComputeActivation(output, output, layer.NbOutputs, activation);
    }

    /// <summary><c>compute_generic_gru</c>: updates <paramref name="state"/> in place.</summary>
    public static void ComputeGenericGru(LinearLayer inputWeights, LinearLayer recurrentWeights, Span<float> state, ReadOnlySpan<float> input)
    {
        int n = recurrentWeights.NbInputs;
        Span<float> zrh = stackalloc float[3 * n];
        Span<float> recur = stackalloc float[3 * n];
        var z = zrh[..n];
        var r = zrh.Slice(n, n);
        var h = zrh.Slice(2 * n, n);
        ComputeLinear(inputWeights, zrh, input);
        ComputeLinear(recurrentWeights, recur, state);
        for (int i = 0; i < 2 * n; i++) zrh[i] += recur[i];
        ComputeActivation(zrh, zrh, 2 * n, Activation.Sigmoid);
        for (int i = 0; i < n; i++) h[i] += recur[2 * n + i] * r[i];
        ComputeActivation(h, h, n, Activation.Tanh);
        for (int i = 0; i < n; i++) h[i] = z[i] * state[i] + (1 - z[i]) * h[i];
        h.CopyTo(state);
    }

    /// <summary><c>compute_glu</c>. <paramref name="output"/> may alias <paramref name="input"/>.</summary>
    public static void ComputeGlu(LinearLayer layer, Span<float> output, ReadOnlySpan<float> input)
    {
        int n = layer.NbOutputs;
        Span<float> act2 = stackalloc float[n];
        ComputeLinear(layer, act2, input);
        ComputeActivation(act2, act2, n, Activation.Sigmoid);
        for (int i = 0; i < n; i++) output[i] = input[i] * act2[i];
    }

    /// <summary><c>compute_generic_conv1d</c>: <paramref name="mem"/> holds the previous (kernel-1) inputs.</summary>
    public static void ComputeGenericConv1d(LinearLayer layer, Span<float> output, Span<float> mem, ReadOnlySpan<float> input, int inputSize, Activation activation)
    {
        int nbIn = layer.NbInputs;
        if (nbIn > MaxConvInputsAll) throw new ArgumentOutOfRangeException(nameof(layer));
        Span<float> tmp = stackalloc float[nbIn];
        if (nbIn != inputSize) mem[..(nbIn - inputSize)].CopyTo(tmp);
        input[..inputSize].CopyTo(tmp[(nbIn - inputSize)..]);
        ComputeLinear(layer, output, tmp);
        ComputeActivation(output, output, layer.NbOutputs, activation);
        if (nbIn != inputSize) tmp.Slice(inputSize, nbIn - inputSize).CopyTo(mem);
    }

    /// <summary><c>compute_generic_conv1d_dilation</c>.</summary>
    public static void ComputeGenericConv1dDilation(LinearLayer layer, Span<float> output, Span<float> mem, ReadOnlySpan<float> input, int inputSize, int dilation, Activation activation)
    {
        int nbIn = layer.NbInputs;
        if (nbIn > MaxConvInputsAll) throw new ArgumentOutOfRangeException(nameof(layer));
        Span<float> tmp = stackalloc float[nbIn];
        int ksize = nbIn / inputSize;
        if (dilation == 1) mem[..(nbIn - inputSize)].CopyTo(tmp);
        else for (int i = 0; i < ksize - 1; i++) mem.Slice(i * inputSize * dilation, inputSize).CopyTo(tmp[(i * inputSize)..]);
        input[..inputSize].CopyTo(tmp[(nbIn - inputSize)..]);
        ComputeLinear(layer, output, tmp);
        ComputeActivation(output, output, layer.NbOutputs, activation);
        if (dilation == 1) tmp.Slice(inputSize, nbIn - inputSize).CopyTo(mem);
        else
        {
            int len = inputSize * dilation * (ksize - 1) - inputSize;
            mem.Slice(inputSize, len).CopyTo(mem);   // overlapping move, like OPUS_COPY/memmove
            input[..inputSize].CopyTo(mem[len..]);
        }
    }

    /// <summary><c>compute_conv2d</c>.</summary>
    public static void ComputeConv2d(Conv2dLayer conv, Span<float> output, Span<float> mem, ReadOnlySpan<float> input, int height, int hstride, Activation activation)
    {
        int timeStride = conv.InChannels * (height + conv.KHeight - 1);
        if (conv.KTime * timeStride > MaxConv2dInputs) throw new ArgumentOutOfRangeException(nameof(conv));
        Span<float> inBuf = stackalloc float[conv.KTime * timeStride];
        mem[..((conv.KTime - 1) * timeStride)].CopyTo(inBuf);
        input[..timeStride].CopyTo(inBuf[((conv.KTime - 1) * timeStride)..]);
        inBuf.Slice(timeStride, (conv.KTime - 1) * timeStride).CopyTo(mem);
        var weights = conv.FloatWeights!;
        if (conv.KHeight == 3 && conv.KTime == 3)
            Conv2d3x3Float(output, weights, conv.InChannels, conv.OutChannels, inBuf, height, hstride);
        else
            Conv2dFloat(output, weights, conv.InChannels, conv.OutChannels, conv.KTime, conv.KHeight, inBuf, height, hstride);
        if (conv.Bias != null)
        {
            for (int i = 0; i < conv.OutChannels; i++)
                for (int j = 0; j < height; j++) output[i * hstride + j] += conv.Bias[i];
        }
        for (int i = 0; i < conv.OutChannels; i++)
        {
            var o = output.Slice(i * hstride, height);
            ComputeActivation(o, o, height, activation);
        }
    }

    private static void Conv2dFloat(Span<float> output, ReadOnlySpan<float> weights, int inChannels, int outChannels, int ktime, int kheight, ReadOnlySpan<float> input, int height, int hstride)
    {
        int inStride = height + kheight - 1;
        for (int i = 0; i < outChannels; i++)
        {
            output.Slice(i * hstride, height).Clear();
            for (int m = 0; m < inChannels; m++)
                for (int t = 0; t < ktime; t++)
                    for (int h = 0; h < kheight; h++)
                        for (int j = 0; j < height; j++)
                            output[i * hstride + j] += weights[i * inChannels * ktime * kheight + m * ktime * kheight + t * kheight + h] *
                                                       input[t * inChannels * inStride + m * inStride + j + h];
        }
    }

    private static void Conv2d3x3Float(Span<float> output, ReadOnlySpan<float> weights, int inChannels, int outChannels, ReadOnlySpan<float> input, int height, int hstride)
    {
        const int k = 3;
        int inStride = height + k - 1;
        for (int i = 0; i < outChannels; i++)
        {
            output.Slice(i * hstride, height).Clear();
            for (int m = 0; m < inChannels; m++)
            {
                var w = weights.Slice(i * inChannels * 9 + m * 9, 9);
                var i0 = input.Slice(0 * inChannels * inStride + m * inStride);
                var i1 = input.Slice(1 * inChannels * inStride + m * inStride);
                var i2 = input.Slice(2 * inChannels * inStride + m * inStride);
                for (int j = 0; j < height; j++)
                {
                    output[i * hstride + j] += w[0] * i0[j] + w[1] * i0[j + 1] + w[2] * i0[j + 2]
                                             + w[3] * i1[j] + w[4] * i1[j + 1] + w[5] * i1[j + 2]
                                             + w[6] * i2[j] + w[7] * i2[j + 1] + w[8] * i2[j + 2];
                }
            }
        }
    }
}
