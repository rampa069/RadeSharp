// SPDX-License-Identifier: BSD-3-Clause
// Port of Opus dnn/fargan.c (Amazon, Jean-Marc Valin): FARGAN neural vocoder,
// 20 features (10 ms frame) -> 160 samples of 16 kHz speech.

using RadeSharp.Models;
using RadeSharp.Nnet;
using static RadeSharp.Nnet.NnetOps;
using static RadeSharp.Vocoder.LpcnetConst;

namespace RadeSharp.Vocoder;

/// <summary><c>FARGANState</c> + fargan_init / fargan_cont / fargan_synthesize.</summary>
public sealed class Fargan
{
    public const int ContSamples = 320;           // FARGAN_CONT_SAMPLES
    public const int NbSubframes = 4;
    public const int SubframeSize = 40;
    public const int FrameSizeSamples = NbSubframes * SubframeSize;   // 160
    private const int CondSize = 320 / NbSubframes;                    // COND_NET_FDENSE2_OUT_SIZE / 4 = 80
    private const float Deemphasis = 0.85f;
    private const int SigNetInputSize = CondSize + 2 * SubframeSize + 4;   // 164
    private const int Fwc0StateSize = 2 * SigNetInputSize;
    private const int PembedOut = 12, Fconv1In = 64, Fconv1Out = 128, Fdense2Out = 320;
    private const int Fwc0ConvOut = 192, Gru1Out = 160, Gru2Out = 128, Gru3Out = 128, SkipDenseOut = 128;

    private readonly FarganModel _model;
    private bool _contInitialized;
    private float _deemphMem;
    private readonly float[] _pitchBuf = new float[PitchMaxPeriod];
    private readonly float[] _condConv1State = new float[64 * 2];
    private readonly float[] _fwc0Mem = new float[Fwc0StateSize];
    private readonly float[] _gru1State = new float[Gru1Out];
    private readonly float[] _gru2State = new float[Gru2Out];
    private readonly float[] _gru3State = new float[Gru3Out];
    private int _lastPeriod;

    /// <summary><c>fargan_init</c> with the built-in weights.</summary>
    public Fargan() => _model = BuiltinWeights.Fargan.Value;

    /// <summary>
    /// Pitch period from the pitch feature: floor(.5 + 256 / 2^((f+1.5)*60/60)).
    /// clang lowers this pow(2.f, x) to exp2(); Math.Pow can differ from exp2 in
    /// the last ulp, which can only matter at an exact .5 rounding boundary.
    /// </summary>
    private static int Period(float pitchFeature) =>
        (int)Math.Floor(.5 + 256.0 / Math.Pow(2.0f, (1.0 / 60.0) * ((pitchFeature + 1.5) * 60)));

    private void ComputeCond(Span<float> cond, ReadOnlySpan<float> features, int period)
    {
        Span<float> denseIn = stackalloc float[NbFeatures + PembedOut];
        Span<float> conv1In = stackalloc float[Fconv1In];
        Span<float> fdense2In = stackalloc float[Fconv1Out];
        int row = Math.Max(0, Math.Min(period - 32, 223));
        _model.cond_net_pembed.FloatWeights.AsSpan(row * PembedOut, PembedOut).CopyTo(denseIn[NbFeatures..]);
        features[..NbFeatures].CopyTo(denseIn);
        ComputeGenericDense(_model.cond_net_fdense1, conv1In, denseIn, Activation.Tanh);
        ComputeGenericConv1d(_model.cond_net_fconv1, fdense2In, _condConv1State, conv1In, Fconv1In, Activation.Tanh);
        ComputeGenericDense(_model.cond_net_fdense2, cond, fdense2In, Activation.Tanh);
    }

    private void Deemph(Span<float> pcm)
    {
        for (int i = 0; i < SubframeSize; i++)
        {
            pcm[i] += Deemphasis * _deemphMem;
            _deemphMem = pcm[i];
        }
    }

    private void RunSubframe(Span<float> pcm, ReadOnlySpan<float> cond, int period)
    {
        var m = _model;
        Span<float> fwc0In = stackalloc float[SigNetInputSize];
        Span<float> gru1In = stackalloc float[Fwc0ConvOut + 2 * SubframeSize];
        Span<float> gru2In = stackalloc float[Gru1Out + 2 * SubframeSize];
        Span<float> gru3In = stackalloc float[Gru2Out + 2 * SubframeSize];
        Span<float> pred = stackalloc float[SubframeSize + 4];
        Span<float> prev = stackalloc float[SubframeSize];
        Span<float> pitchGate = stackalloc float[4];
        Span<float> gainBuf = stackalloc float[1];
        Span<float> skipCat = stackalloc float[Gru1Out + Gru2Out + Gru3Out + Fwc0ConvOut + 2 * SubframeSize];
        Span<float> skipOut = stackalloc float[SkipDenseOut];

        ComputeGenericDense(m.sig_net_cond_gain_dense, gainBuf, cond, Activation.Linear);
        float gain = (float)Math.Exp(gainBuf[0]);
        float gain1 = 1.0f / (1e-5f + gain);

        int pos = PitchMaxPeriod - period - 2;
        for (int i = 0; i < SubframeSize + 4; i++)
        {
            float v = gain1 * _pitchBuf[Math.Max(0, pos)];
            float mx = -1.0f > v ? -1.0f : v;            // MAX32(-1.f, v)
            pred[i] = 1.0f < mx ? 1.0f : mx;              // MIN32(1.f, .)
            pos++;
            if (pos == PitchMaxPeriod) pos -= period;
        }
        for (int i = 0; i < SubframeSize; i++)
        {
            float v = gain1 * _pitchBuf[PitchMaxPeriod - SubframeSize + i];
            float mn = 1.0f < v ? 1.0f : v;               // MIN16(1.f, v)
            prev[i] = -1.0f > mn ? -1.0f : mn;            // MAX32(-1.f, .)
        }

        cond[..CondSize].CopyTo(fwc0In);
        pred.CopyTo(fwc0In[CondSize..]);
        prev.CopyTo(fwc0In[(CondSize + SubframeSize + 4)..]);

        ComputeGenericConv1d(m.sig_net_fwc0_conv, gru1In, _fwc0Mem, fwc0In, SigNetInputSize, Activation.Tanh);
        ComputeGlu(m.sig_net_fwc0_glu_gate, gru1In, gru1In);
        ComputeGenericDense(m.sig_net_gain_dense_out, pitchGate, gru1In, Activation.Sigmoid);

        for (int i = 0; i < SubframeSize; i++) gru1In[Fwc0ConvOut + i] = pitchGate[0] * pred[i + 2];
        prev.CopyTo(gru1In[(Fwc0ConvOut + SubframeSize)..]);
        ComputeGenericGru(m.sig_net_gru1_input, m.sig_net_gru1_recurrent, _gru1State, gru1In);
        ComputeGlu(m.sig_net_gru1_glu_gate, gru2In, _gru1State);

        for (int i = 0; i < SubframeSize; i++) gru2In[Gru1Out + i] = pitchGate[1] * pred[i + 2];
        prev.CopyTo(gru2In[(Gru1Out + SubframeSize)..]);
        ComputeGenericGru(m.sig_net_gru2_input, m.sig_net_gru2_recurrent, _gru2State, gru2In);
        ComputeGlu(m.sig_net_gru2_glu_gate, gru3In, _gru2State);

        for (int i = 0; i < SubframeSize; i++) gru3In[Gru2Out + i] = pitchGate[2] * pred[i + 2];
        prev.CopyTo(gru3In[(Gru2Out + SubframeSize)..]);
        ComputeGenericGru(m.sig_net_gru3_input, m.sig_net_gru3_recurrent, _gru3State, gru3In);
        ComputeGlu(m.sig_net_gru3_glu_gate, skipCat[(Gru1Out + Gru2Out)..], _gru3State);

        gru2In[..Gru1Out].CopyTo(skipCat);
        gru3In[..Gru2Out].CopyTo(skipCat[Gru1Out..]);
        int o = Gru1Out + Gru2Out + Gru3Out;
        gru1In[..Fwc0ConvOut].CopyTo(skipCat[o..]);
        for (int i = 0; i < SubframeSize; i++) skipCat[o + Fwc0ConvOut + i] = pitchGate[3] * pred[i + 2];
        prev.CopyTo(skipCat[(o + Fwc0ConvOut + SubframeSize)..]);

        ComputeGenericDense(m.sig_net_skip_dense, skipOut, skipCat, Activation.Tanh);
        ComputeGlu(m.sig_net_skip_glu_gate, skipOut, skipOut);
        ComputeGenericDense(m.sig_net_sig_dense_out, pcm, skipOut, Activation.Tanh);
        for (int i = 0; i < SubframeSize; i++) pcm[i] *= gain;

        Array.Copy(_pitchBuf, SubframeSize, _pitchBuf, 0, PitchMaxPeriod - SubframeSize);
        pcm[..SubframeSize].CopyTo(_pitchBuf.AsSpan(PitchMaxPeriod - SubframeSize));
        Deemph(pcm);
    }

    /// <summary>
    /// <c>fargan_cont</c>: primes the vocoder with <see cref="ContSamples"/> of past speech
    /// and 5 feature vectors laid out every <see cref="LpcnetConst.NbFeatures"/> floats.
    /// </summary>
    public void Cont(ReadOnlySpan<float> pcm0, ReadOnlySpan<float> features0)
    {
        Span<float> cond = stackalloc float[Fdense2Out];
        Span<float> x0 = stackalloc float[ContSamples];
        Span<float> dummy = stackalloc float[SubframeSize];
        int period = 0;
        for (int i = 0; i < 5; i++)
        {
            var features = features0[(i * NbFeatures)..];
            _lastPeriod = period;
            period = Period(features[NbBands]);
            ComputeCond(cond, features, period);
        }
        x0[0] = 0;
        for (int i = 1; i < ContSamples; i++) x0[i] = pcm0[i] - Deemphasis * pcm0[i - 1];
        x0[..FrameSizeSamples].CopyTo(_pitchBuf.AsSpan(PitchMaxPeriod - FrameSizeSamples));
        _contInitialized = true;
        for (int i = 0; i < NbSubframes; i++)
        {
            RunSubframe(dummy, cond.Slice(i * CondSize, CondSize), _lastPeriod);
            x0.Slice(FrameSizeSamples + i * SubframeSize, SubframeSize).CopyTo(_pitchBuf.AsSpan(PitchMaxPeriod - SubframeSize));
        }
        _deemphMem = pcm0[ContSamples - 1];
    }

    /// <summary><c>fargan_synthesize</c>: one feature vector -> <see cref="FrameSizeSamples"/> float samples.</summary>
    public void Synthesize(Span<float> pcm, ReadOnlySpan<float> features)
    {
        if (!_contInitialized) throw new InvalidOperationException("call Cont() first");
        Span<float> cond = stackalloc float[Fdense2Out];
        int period = Period(features[NbBands]);
        ComputeCond(cond, features, period);
        for (int sf = 0; sf < NbSubframes; sf++)
            RunSubframe(pcm.Slice(sf * SubframeSize, SubframeSize), cond.Slice(sf * CondSize, CondSize), _lastPeriod);
        _lastPeriod = period;
    }

    /// <summary><c>fargan_synthesize_int</c>.</summary>
    public void SynthesizeInt(Span<short> pcm, ReadOnlySpan<float> features)
    {
        Span<float> fpcm = stackalloc float[FrameSizeSamples];
        Synthesize(fpcm, features);
        for (int i = 0; i < LpcnetFrameSize; i++) pcm[i] = ToInt16(fpcm[i]);
    }

    /// <summary>(int)floor(.5 + MIN32(32767, MAX32(-32767, 32768.f*x))), as lpcnet_demo and fargan_synthesize_int.</summary>
    public static short ToInt16(float x)
    {
        float v = 32768.0f * x;
        float mx = -32767 > v ? -32767 : v;
        float mn = 32767 < mx ? 32767 : mx;
        return (short)(int)Math.Floor(.5 + mn);
    }
}
