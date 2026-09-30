// SPDX-License-Identifier: BSD-3-Clause
// Port of Opus dnn/pitchdnn.c (Amazon): neural pitch estimator used by the
// LPCNet feature analysis.

using RadeSharp.Models;
using static RadeSharp.Nnet.NnetOps;
using static RadeSharp.Vocoder.LpcnetConst;
using Activation = RadeSharp.Nnet.Activation;

namespace RadeSharp.Vocoder;

internal sealed class PitchDnn
{
    private readonly PitchDnnModel _model = BuiltinWeights.PitchDnn.Value;
    private readonly float[] _gruState = new float[64];
    private readonly float[] _xcorrMem1 = new float[(NbXcorrFeatures + 2) * 2];
    private readonly float[] _xcorrMem2 = new float[(NbXcorrFeatures + 2) * 2 * 8];

    /// <summary><c>compute_pitchdnn</c>: returns the pitch feature (log-period domain).</summary>
    public float Compute(ReadOnlySpan<float> ifFeatures, ReadOnlySpan<float> xcorrFeatures)
    {
        var m = _model;
        Span<float> if1Out = stackalloc float[64];
        Span<float> downsamplerIn = stackalloc float[NbXcorrFeatures + 64];
        Span<float> downsamplerOut = stackalloc float[64];
        Span<float> conv1Tmp1 = stackalloc float[(NbXcorrFeatures + 2) * 8];
        Span<float> conv1Tmp2 = stackalloc float[(NbXcorrFeatures + 2) * 8];
        Span<float> output = stackalloc float[192];
        conv1Tmp1.Clear();
        conv1Tmp2.Clear();

        ComputeGenericDense(m.dense_if_upsampler_1, if1Out, ifFeatures, Activation.Tanh);
        ComputeGenericDense(m.dense_if_upsampler_2, downsamplerIn[NbXcorrFeatures..], if1Out, Activation.Tanh);
        xcorrFeatures[..NbXcorrFeatures].CopyTo(conv1Tmp1[1..]);
        ComputeConv2d(m.conv2d_1, conv1Tmp2[1..], _xcorrMem1, conv1Tmp1, NbXcorrFeatures, NbXcorrFeatures + 2, Activation.Tanh);
        ComputeConv2d(m.conv2d_2, downsamplerIn, _xcorrMem2, conv1Tmp2, NbXcorrFeatures, NbXcorrFeatures, Activation.Tanh);
        ComputeGenericDense(m.dense_downsampler, downsamplerOut, downsamplerIn, Activation.Tanh);
        ComputeGenericGru(m.gru_1_input, m.gru_1_recurrent, _gruState, downsamplerOut);
        ComputeGenericDense(m.dense_final_upsampler, output, _gruState, Activation.Linear);

        int pos = 0;
        float maxval = -1, sum = 0, count = 0;
        for (int i = 0; i < 180; i++)
        {
            if (output[i] > maxval)
            {
                pos = i;
                maxval = output[i];
            }
        }
        for (int i = Math.Max(0, pos - 2); i <= Math.Min(179, pos + 2); i++)
        {
            float p = (float)Math.Exp(output[i]);
            sum += p * i;
            count += p;
        }
        return (float)((1.0f / 60.0f) * (sum / count) - 1.5);
    }
}
