// SPDX-License-Identifier: BSD-3-Clause
// Port of Opus dnn/lpcnet_enc.c (Mozilla, Amazon): LPCNet/FARGAN feature
// analysis, 160 samples of 16 kHz speech -> 36 floats (20 used by RADE/FARGAN).

using RadeSharp.Nnet;
using static RadeSharp.Vocoder.LpcnetConst;

namespace RadeSharp.Vocoder;

/// <summary><c>LPCNetEncState</c> + lpcnet_compute_single_frame_features.</summary>
public sealed class LpcnetEncoder
{
    private static readonly Lazy<WeightSet> Tables = new(() =>
    {
        using var s = typeof(LpcnetEncoder).Assembly.GetManifestResourceStream("RadeSharp.Weights.lpcnet_tables.bin")
            ?? throw new InvalidOperationException("embedded lpcnet_tables.bin missing");
        var data = new byte[s.Length];
        s.ReadExactly(data);
        return WeightSet.FromBlob(data);
    });

    private static readonly float[] LpB = [-0.84946f, 1f];   // [b,a]=ellip(2, 2, 20, 1200/8000)
    private static readonly float[] LpA = [-1.54220f, 0.70781f];

    private readonly Freq _freq = new(Tables.Value);
    private readonly PitchDnn _pitchdnn = new();
    private readonly float[] _analysisMem = new float[OverlapSize];
    private float _memPreemph;
    private readonly KissCpx[] _prevIf = new KissCpx[PitchIfMaxFreq];
    private readonly float[] _ifFeatures = new float[PitchIfFeatures];
    private readonly float[] _xcorrFeatures = new float[PitchMaxPeriod - PitchMinPeriod];
    private float _dnnPitch;
    private readonly float[] _pitchMem = new float[LpcOrder];
    private float _pitchFilt;
    private readonly float[] _excBuf = new float[PitchBufSize];
    private readonly float[] _lpBuf = new float[PitchBufSize];
    private readonly float[] _lpMem = new float[4];
    private readonly float[] _lpc = new float[LpcOrder];
    private readonly float[] _features = new float[NbTotalFeatures];

    public const int FrameSamples = FrameSize;          // 160 @ 16 kHz
    public const int FeaturesPerFrame = NbTotalFeatures; // 36

    private void FrameAnalysis(Span<KissCpx> X, Span<float> ex, ReadOnlySpan<float> input)
    {
        Span<float> x = stackalloc float[WindowSize];
        _analysisMem.CopyTo(x);
        input[..FrameSize].CopyTo(x[OverlapSize..]);
        input.Slice(FrameSize - OverlapSize, OverlapSize).CopyTo(_analysisMem);
        _freq.ApplyWindow(x);
        _freq.ForwardTransform(X, x);
        Freq.ComputeBandEnergy(ex, X);
    }

    /// <summary><c>biquad</c> (reordered form from lpcnet_enc.c); y may alias x.</summary>
    private static void Biquad(Span<float> y, Span<float> mem, ReadOnlySpan<float> x, ReadOnlySpan<float> b, ReadOnlySpan<float> a, int n)
    {
        float mem0 = mem[0], mem1 = mem[1];
        for (int i = 0; i < n; i++)
        {
            float xi = x[i];
            float yi = x[i] + mem0;
            float mem00 = mem0;
            mem0 = (b[0] - a[0]) * xi + mem1 - a[0] * mem0;
            mem1 = (b[1] - a[1]) * xi + 1e-30f - a[1] * mem00;
            y[i] = yi;
        }
        mem[0] = mem0;
        mem[1] = mem1;
    }

    private static float Clamp1(float v)
    {
        float mn = 1.0f < v ? 1.0f : v;       // MIN16(1.f, v)
        return -1.0f > mn ? -1.0f : mn;       // MAX16(-1.f, .)
    }

    private static float Max16(float a, float b) => a > b ? a : b;

    /// <summary><c>compute_frame_features</c>.</summary>
    private void ComputeFrameFeatures(ReadOnlySpan<float> input)
    {
        Span<float> alignedIn = stackalloc float[FrameSize];
        Span<float> ly = stackalloc float[NbBands];
        Span<KissCpx> X = stackalloc KissCpx[FreqSize];
        Span<float> ex = stackalloc float[NbBands];
        Span<float> xcorr = stackalloc float[PitchMaxPeriod];
        Span<float> x = stackalloc float[FrameSize + LpcOrder];
        Span<float> enerNorm = stackalloc float[PitchMaxPeriod - PitchMinPeriod];

        _analysisMem.AsSpan(OverlapSize - TrainingOffset, TrainingOffset).CopyTo(alignedIn);
        FrameAnalysis(X, ex, input);
        _ifFeatures[0] = Clamp1((1.0f / 64) * (10.0f * CeltMath.Log10(1e-15f + X[0].R * X[0].R) - 6.0f));
        for (int i = 1; i < PitchIfMaxFreq; i++)
        {
            // C_MULC(prod, X[i], prev_if[i])
            KissCpx a = X[i], b = _prevIf[i];
            KissCpx prod = new(a.R * b.R + a.I * b.I, a.I * b.R - a.R * b.I);
            float norm1 = (float)(1.0f / Math.Sqrt(1e-15f + prod.R * prod.R + prod.I * prod.I));
            prod = new(prod.R * norm1, prod.I * norm1);
            _ifFeatures[3 * i - 2] = prod.R;
            _ifFeatures[3 * i - 1] = prod.I;
            _ifFeatures[3 * i] = Clamp1((1.0f / 64) * (10.0f * CeltMath.Log10(1e-15f + X[i].R * X[i].R + X[i].I * X[i].I) - 6.0f));
        }
        X[..PitchIfMaxFreq].CopyTo(_prevIf);

        float logMax = -2, follow = -2;
        for (int i = 0; i < NbBands; i++)
        {
            ly[i] = CeltMath.Log10(1e-2f + ex[i]);
            ly[i] = Max16(logMax - 8, Max16(follow - 2.5f, ly[i]));
            logMax = Max16(logMax, ly[i]);
            follow = Max16(follow - 2.5f, ly[i]);
        }
        _freq.Dct(_features, ly);
        _features[0] -= 4;
        _freq.LpcFromCepstrum(_lpc, _features);
        for (int i = 0; i < LpcOrder; i++) _features[NbBands + 2 + i] = _lpc[i];

        Array.Copy(_excBuf, FrameSize, _excBuf, 0, PitchMaxPeriod);
        Array.Copy(_lpBuf, FrameSize, _lpBuf, 0, PitchMaxPeriod);
        input[..(FrameSize - TrainingOffset)].CopyTo(alignedIn[TrainingOffset..]);
        _pitchMem.CopyTo(x);
        alignedIn.CopyTo(x[LpcOrder..]);
        alignedIn.Slice(FrameSize - LpcOrder, LpcOrder).CopyTo(_pitchMem);
        CeltMath.Fir(x, _lpc, _lpBuf.AsSpan(PitchMaxPeriod), FrameSize, LpcOrder);
        for (int i = 0; i < FrameSize; i++)
        {
            _excBuf[PitchMaxPeriod + i] = _lpBuf[PitchMaxPeriod + i] + .7f * _pitchFilt;
            _pitchFilt = _lpBuf[PitchMaxPeriod + i];
        }
        var lp = _lpBuf.AsSpan(PitchMaxPeriod, FrameSize);
        Biquad(lp, _lpMem, lp, LpB, LpA, FrameSize);

        {
            var buf = _excBuf.AsSpan();
            CeltMath.PitchXcorr(buf[PitchMaxPeriod..], buf, xcorr, FrameSize, PitchMaxPeriod - PitchMinPeriod);
            float ener0 = CeltMath.InnerProd(buf[PitchMaxPeriod..], buf[PitchMaxPeriod..], FrameSize);
            double ener1 = CeltMath.InnerProd(buf, buf, FrameSize);
            for (int i = 0; i < PitchMaxPeriod - PitchMinPeriod; i++)
            {
                float ener = (float)(1 + ener0 + ener1);
                _xcorrFeatures[i] = 2 * xcorr[i];
                enerNorm[i] = ener;
                ener1 += buf[i + FrameSize] * (double)buf[i + FrameSize] - buf[i] * (double)buf[i];
            }
            for (int i = 0; i < PitchMaxPeriod - PitchMinPeriod; i++) _xcorrFeatures[i] /= enerNorm[i];
        }

        _dnnPitch = _pitchdnn.Compute(_ifFeatures, _xcorrFeatures);
        // C lowers pow(2.f, x) to exp2(); see Fargan.Period.
        int pitch = (int)Math.Floor(.5 + 256.0 / Math.Pow(2.0f, (1.0 / 60.0) * ((_dnnPitch + 1.5) * 60)));
        float xx = CeltMath.InnerProd(_lpBuf.AsSpan(PitchMaxPeriod), _lpBuf.AsSpan(PitchMaxPeriod), FrameSize);
        float yy = CeltMath.InnerProd(_lpBuf.AsSpan(PitchMaxPeriod - pitch), _lpBuf.AsSpan(PitchMaxPeriod - pitch), FrameSize);
        float xy = CeltMath.InnerProd(_lpBuf.AsSpan(PitchMaxPeriod), _lpBuf.AsSpan(PitchMaxPeriod - pitch), FrameSize);
        float frameCorr = (float)(xy / Math.Sqrt(1 + xx * yy));
        frameCorr = (float)(Math.Log(1.0f + Math.Exp(5.0f * frameCorr)) / Math.Log(1 + Math.Exp(5.0f)));
        _features[NbBands] = _dnnPitch;
        _features[NbBands + 1] = frameCorr - .5f;
    }

    /// <summary><c>lpcnet_compute_single_frame_features</c>: 160 int16 samples -> 36 features.</summary>
    public void ComputeFeatures(ReadOnlySpan<short> pcm, Span<float> features)
    {
        Span<float> x = stackalloc float[FrameSize];
        for (int i = 0; i < FrameSize; i++) x[i] = pcm[i];
        ComputeImpl(x, features);
    }

    /// <summary><c>lpcnet_compute_single_frame_features_float</c> (input in int16 scale).</summary>
    public void ComputeFeatures(ReadOnlySpan<float> pcm, Span<float> features)
    {
        Span<float> x = stackalloc float[FrameSize];
        pcm[..FrameSize].CopyTo(x);
        ComputeImpl(x, features);
    }

    private void ComputeImpl(Span<float> x, Span<float> features)
    {
        // preemphasis(x, &mem_preemph, x, PREEMPHASIS, FRAME_SIZE)
        for (int i = 0; i < FrameSize; i++)
        {
            float yi = x[i] + _memPreemph;
            _memPreemph = -Preemphasis * x[i];
            x[i] = yi;
        }
        ComputeFrameFeatures(x);
        _features.AsSpan(0, NbTotalFeatures).CopyTo(features);
    }
}
