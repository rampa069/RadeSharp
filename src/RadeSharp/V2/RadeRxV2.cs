// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_rx_v2.c (Copyright (C) 2025 David Rowe): RADE V2 receiver
// (CP autocorrelation acquisition, FrameSyncNet, channel-sparsity EOO, AGC).

using System.Globalization;
using RadeSharp.Core;
using RadeSharp.Dsp;
using RadeSharp.Models;
using static RadeSharp.Dsp.RadeDsp;
using static RadeSharp.V2.OfdmV2;

namespace RadeSharp.V2;

internal sealed class RadeRxV2
{
    public const int StateIdle = 0;
    public const int StateSync = 1;
    public const int RxBufSize = 3 * SymLen;
    public const int FeaturesOut = RadeTxV2.FramesPerStepV2 * NbTotalFeaturesV2;

    private const float Alpha = 0.95f;      // Ry_smooth IIR coefficient
    private const float Beta = 0.999f;      // delta_hat / freq_offset IIR coefficient
    private const float Tsig = 0.38f;       // signal-detection threshold on |Ry_smooth|
    private const float Tsin = 4.0f;        // sine-wave detection ratio threshold
    private const float Teoo = 0.75f;       // EOO smoothed sparsity threshold
    private const float AlphaEoo = 0.70f;   // EOO smoother IIR coefficient
    private const float SnrCorrA = 1.24392558f;
    private const float SnrCorrB = 3.33253932f;
    private const int TimingShift = SymLen / 4;
    private const float AgcAlpha = 0.99875f; // AGC power IIR, tau ~0.1 s @ 8 kHz
    private const float PiF = (float)Math.PI;

    private readonly OfdmV2 _ofdm = new();
    private readonly Bpf? _bpf;
    private readonly RadeDecV2Model _decModel = BuiltinWeights.RadeDecV2.Value;
    private readonly RadeDecoderV2 _decState = new();
    private readonly RadeSyncModel _syncModel = BuiltinWeights.RadeSync.Value;

    public int State = StateIdle;
    private int _count, _count1, _s, _i;
    private readonly bool _timingAdj = true;
    private int _nAcq;
    private readonly int _hangover = 75;

    public bool AgcEn = true;
    private readonly float _agcTarget;
    private float _agcPower;
    public float Gain;

    public float DeltaHat, DeltaHatG, FreqOffset, FreqOffsetG;
    private float _ryMax, _ryMin;
    private bool _newSigDeltaHat, _newSigFHat;

    public float SnrEstDb;
    private readonly float _snrOffsetDb;
    private readonly float _snrCorrA = SnrCorrA, _snrCorrB = SnrCorrB;

    private float _frameSyncEven, _frameSyncOdd;
    private float _eooSmooth;

    private readonly RadeComp[] _rxBuf = new RadeComp[RxBufSize];
    private readonly RadeComp[] _rxI = new RadeComp[2 * SymLen];
    private readonly RadeComp[] _rxSymTd = new RadeComp[MV2];
    private RadeComp _rxPhase = new(1.0f, 0.0f);
    private readonly RadeComp[] _ryNorm = new RadeComp[SymLen];
    private readonly RadeComp[] _rySmooth = new RadeComp[SymLen];
    private readonly float[] _azHat = new float[RadeTxV2.LatentDimV2];
    private readonly RadeComp[] _rxFiltered = new RadeComp[SymLen + TimingShift];
    private readonly RadeComp[] _rxScaled = new RadeComp[SymLen + TimingShift];

    private float _dataSymbol;
    private int _nin = SymLen;
    public int Verbose;

    /// <summary><c>rade_rx_v2_init</c>.</summary>
    public RadeRxV2(bool bpfEn)
    {
        if (bpfEn)
        {
            float w0 = _ofdm.W[0];
            float wN = _ofdm.W[NcV2 - 1];
            float bandwidth = 1.2f * (wN - w0) * (float)Fs / (2.0f * PiF);
            float centre = (wN + w0) * (float)Fs / (2.0f * PiF) / 2.0f;
            _bpf = new Bpf(BpfNtap, Fs, bandwidth, centre, SymLen + BpfNtap);
            _snrOffsetDb = 10.0f * MathF.Log10(3000.0f / (bandwidth + 1e-12f));
        }
        else
        {
            float rsDash = (float)Fs / MV2;
            float bandwidth = 1.2f * rsDash * (NcV2 - 1);
            _snrOffsetDb = 10.0f * MathF.Log10(3000.0f / (bandwidth + 1e-12f));
        }
        // AGC target: nominal peak 1.0 backed off by the ~3 dB PAPR (radae_v2.py agc_target)
        _agcTarget = 1.0f * MathF.Pow(10.0f, -3.0f / 20.0f);
        _agcPower = _agcTarget * _agcTarget;
    }

    public int Nin => _nin;
    public static int NinMax => SymLen + TimingShift;
    public float DataSymbol => _dataSymbol;

    private void ComputeAutocorr()
    {
        for (int gamma = 0; gamma < SymLen; gamma++)
        {
            int idx = SymLen + gamma;
            RadeComp ry = default;
            float dCp = 0.0f, dM = 0.0f;
            for (int k = 0; k < NcpV2; k++)
            {
                RadeComp a = _rxBuf[idx - NcpV2 + k];
                RadeComp b = _rxBuf[idx - NcpV2 + MV2 + k];
                ry.Real += a.Real * b.Real + a.Imag * b.Imag;
                ry.Imag += a.Imag * b.Real - a.Real * b.Imag;
                dCp += a.Real * a.Real + a.Imag * a.Imag;
                dM += b.Real * b.Real + b.Imag * b.Imag;
            }
            float d = dCp + dM + 1e-12f;
            _ryNorm[gamma] = new(2.0f * ry.Real / d, 2.0f * ry.Imag / d);
            _rySmooth[gamma] = new(Alpha * _rySmooth[gamma].Real + (1.0f - Alpha) * _ryNorm[gamma].Real,
                                   Alpha * _rySmooth[gamma].Imag + (1.0f - Alpha) * _ryNorm[gamma].Imag);
        }
    }

    private void DetectSignal(out bool sigDet, out bool sineDet)
    {
        float maxVal = -1.0f, minVal = 1e30f;
        int maxIdx = 0;
        for (int g = 0; g < SymLen; g++)
        {
            float mag = MathF.Sqrt(_rySmooth[g].Real * _rySmooth[g].Real + _rySmooth[g].Imag * _rySmooth[g].Imag);
            if (mag > maxVal) { maxVal = mag; maxIdx = g; }
            if (mag < minVal) minVal = mag;
        }
        DeltaHatG = maxIdx;
        _ryMax = maxVal;
        _ryMin = minVal;
        sigDet = maxVal > Tsig;
        sineDet = maxVal / (minVal + 1e-12f) < Tsin;

        float rho = maxVal;
        if (rho >= 1.0f) rho = 1.0f - 1e-6f;
        if (rho <= 0.0f) rho = 1e-6f;
        float snrRaw = 10.0f * MathF.Log10(rho / (1.0f - rho)) - _snrOffsetDb;
        SnrEstDb = _snrCorrA * snrRaw + _snrCorrB;
    }

    private void ExtractSymbol()
    {
        int deltaHatRx = (int)DeltaHat - NcpV2;
        float omega = 2.0f * PiF * FreqOffset / (float)Fs;
        Array.Copy(_rxI, SymLen, _rxI, 0, SymLen);
        int st = SymLen + deltaHatRx;
        var (sin, cos) = MathF.SinCos(-omega);
        RadeComp pstep = new(cos, sin);
        for (int n = 0; n < SymLen; n++)
        {
            _rxPhase = Cmul(_rxPhase, pstep);
            _rxI[SymLen + n] = Cmul(_rxPhase, _rxBuf[st + n]);
            if (n >= NcpV2) _rxSymTd[n - NcpV2] = _rxI[SymLen + n];
        }
        float pmag = MathF.Sqrt(_rxPhase.Real * _rxPhase.Real + _rxPhase.Imag * _rxPhase.Imag);
        _rxPhase = new(_rxPhase.Real / pmag, _rxPhase.Imag / pmag);
    }

    private bool UpdateFrameSyncDecode(Span<float> featuresOut)
    {
        float metric = RadeFrameSync.Run(_syncModel, _azHat);
        bool winning;
        if (_s % 2 != 0)
        {
            _frameSyncOdd = Beta * _frameSyncOdd + (1.0f - Beta) * metric;
            winning = _frameSyncOdd > _frameSyncEven;
        }
        else
        {
            _frameSyncEven = Beta * _frameSyncEven + (1.0f - Beta) * metric;
            winning = _frameSyncEven > _frameSyncOdd;
        }
        if (!winning) return false;

        const int frames = RadeTxV2.FramesPerStepV2, numFeat = RadeTxV2.NumFeaturesV2;
        Span<float> decFeatures = stackalloc float[frames * numFeat];
        _decState.Run(_decModel, decFeatures, _azHat);
        featuresOut[..FeaturesOut].Clear();
        for (int f = 0; f < frames; f++)
        {
            var dst = featuresOut.Slice(f * NbTotalFeaturesV2, NbTotalFeaturesV2);
            var src = decFeatures.Slice(f * numFeat, numFeat);
            if (src[18] < -1.4f) src[18] = -1.4f;   // limit_pitch, matches radae_v2.py default
            for (int j = 0; j < NumUsedFeaturesV2; j++) dst[j] = src[j];
        }
        _dataSymbol = decFeatures[NumUsedFeaturesV2];
        return true;
    }

    private bool DetectEoo()
    {
        float metric = _ofdm.EooMetric(_rxSymTd);
        _eooSmooth = AlphaEoo * _eooSmooth + (1.0f - AlphaEoo) * metric;
        return _eooSmooth > Teoo;
    }

    private int AdjustTiming()
    {
        if (!_timingAdj) return SymLen;
        int nin = SymLen;
        Span<RadeComp> tmp = stackalloc RadeComp[TimingShift];
        if (DeltaHat > (float)(3 * SymLen / 4))
        {
            DeltaHat -= TimingShift;
            _rySmooth.AsSpan(0, TimingShift).CopyTo(tmp);
            Array.Copy(_rySmooth, TimingShift, _rySmooth, 0, SymLen - TimingShift);
            tmp.CopyTo(_rySmooth.AsSpan(SymLen - TimingShift));
            nin = SymLen + TimingShift;
        }
        else if (DeltaHat < (float)(SymLen / 4))
        {
            DeltaHat += TimingShift;
            _rySmooth.AsSpan(SymLen - TimingShift, TimingShift).CopyTo(tmp);
            Array.Copy(_rySmooth, 0, _rySmooth, TimingShift, SymLen - TimingShift);
            tmp.CopyTo(_rySmooth);
            nin = SymLen - TimingShift;
        }
        return nin;
    }

    private float ComputeGain(ReadOnlySpan<RadeComp> rxIn, int nin)
    {
        if (!AgcEn) return 1.0f;
        float p = _agcPower;
        for (int i = 0; i < nin; i++)
        {
            float magSq = rxIn[i].Real * rxIn[i].Real + rxIn[i].Imag * rxIn[i].Imag;
            p = AgcAlpha * p + (1.0f - AgcAlpha) * magSq;
        }
        _agcPower = p;
        float gain = _agcTarget / (MathF.Sqrt(p) + 1e-6f);
        if (gain < 0.1f) gain = 0.1f;
        if (gain > 10.0f) gain = 10.0f;
        return gain;
    }

    /// <summary><c>rade_rx_v2_process</c>: returns 0x1 when features are valid, 0x2 on End-of-Over.</summary>
    public int Process(Span<float> featuresOut, ReadOnlySpan<RadeComp> rxIn)
    {
        int nin = _nin;
        ReadOnlySpan<RadeComp> rxSamples = rxIn;
        if (_bpf != null)
        {
            _bpf.Process(_rxFiltered, rxIn, nin);
            rxSamples = _rxFiltered;
        }

        float gain = ComputeGain(rxSamples, nin);
        Gain = gain;
        if (gain != 1.0f)
        {
            for (int i = 0; i < nin; i++) _rxScaled[i] = new(rxSamples[i].Real * gain, rxSamples[i].Imag * gain);
            rxSamples = _rxScaled;
        }

        Array.Copy(_rxBuf, nin, _rxBuf, 0, RxBufSize - nin);
        rxSamples[..nin].CopyTo(_rxBuf.AsSpan(RxBufSize - nin));

        ComputeAutocorr();
        DetectSignal(out bool sigDet, out bool sineDet);

        bool validOutput = false, eooFlag = false;
        int nextState = State;

        if (State == StateIdle)
        {
            if (sigDet && !sineDet) _count++;
            else _count = 0;
            if (_count == 5)
            {
                int dg = (int)DeltaHatG;
                float deltaPhi = MathF.Atan2(_rySmooth[dg].Imag, _rySmooth[dg].Real);
                DeltaHat = DeltaHatG;
                FreqOffset = -deltaPhi * (float)Fs / (2.0f * PiF * MV2);
                _count = 0;
                _count1 = 0;
                _frameSyncEven = 0.0f;
                _frameSyncOdd = 0.0f;
                _eooSmooth = 0.0f;
                _nAcq++;
                nextState = StateSync;
                if (Verbose != 0)
                    RadeLog.Write(string.Format(CultureInfo.InvariantCulture,
                        "sync: acquired n_acq={0} delta_hat={1:F0} freq_offset={2:F2} Hz\n", _nAcq, DeltaHat, FreqOffset));
            }
        }
        else
        {
            int dg = (int)DeltaHatG;
            float deltaPhi = MathF.Atan2(_rySmooth[dg].Imag, _rySmooth[dg].Real);
            FreqOffsetG = -deltaPhi * (float)Fs / (2.0f * PiF * MV2);
            DeltaHat = Beta * DeltaHat + (1.0f - Beta) * (float)DeltaHatG;
            FreqOffset = Beta * FreqOffset + (1.0f - Beta) * FreqOffsetG;

            if (!sigDet || sineDet) _count++;
            else _count = 0;
            if (_count == _hangover)
            {
                nextState = StateIdle;
                _count = 0;
                _count1 = 0;
            }

            // Re-acquire check: new signal with different timing or frequency
            _newSigDeltaHat = MathF.Abs((float)DeltaHatG - DeltaHat) > (float)NcpV2;
            _newSigFHat = MathF.Abs(FreqOffsetG - FreqOffset) > 5.0f;
            if (sigDet && (_newSigDeltaHat || _newSigFHat)) _count1++;
            else _count1 = 0;
            if (_count1 == 5)
            {
                nextState = StateIdle;
                _count = 0;
                _count1 = 0;
            }

            ExtractSymbol();
            _ofdm.DemodFrame(_azHat, _rxI, -16);

            if (DetectEoo())
            {
                if (Verbose != 0) RadeLog.Write("sync: EOO detected\n");
                _count = 0;
                _count1 = 0;
                _eooSmooth = 0.0f;
                Array.Clear(_rySmooth);   // prevent instant re-sync
                nextState = StateIdle;
                eooFlag = true;
            }
            else
            {
                validOutput = UpdateFrameSyncDecode(featuresOut);
                if (validOutput) _i++;
            }

            _nin = AdjustTiming();
        }

        _s++;
        if (Verbose >= 2) LogStatus(sigDet, sineDet);

        State = nextState;
        if (State == StateIdle) _nin = SymLen;
        return (validOutput ? 0x1 : 0) | (eooFlag ? 0x2 : 0);
    }

    private void LogStatus(bool sigDet, bool sineDet)
    {
        if (RadeLog.Writer == null) return;
        var ci = CultureInfo.InvariantCulture;
        string st = State == StateIdle ? "idle" : "sync";
        if (Verbose >= 3)
            RadeLog.Write(string.Format(ci,
                "{0,4} {1,4} {2} nin: {3,3} sig: {4} sine: {5} c: {6,2} nsd: {7} nsf: {8} c1: {9,2} fs: {10} delta_hat: {11,3:F0} g: {12,3:F0} f_off: {13,5:F2} f_off_g: {14,5:F2} Ry_max: {15,5:F2} snr: {16,5:F1} dB eoo: {17:F3}\n",
                _s, _i, st, _nin, sigDet ? 1 : 0, sineDet ? 1 : 0, _count, _newSigDeltaHat ? 1 : 0, _newSigFHat ? 1 : 0, _count1,
                _frameSyncOdd > _frameSyncEven ? 1 : 0, DeltaHat, DeltaHatG, FreqOffset, FreqOffsetG, _ryMax, SnrEstDb, _eooSmooth));
        else
            RadeLog.Write(string.Format(ci, "{0,4} {1,4} {2} sig: {3} f_off: {4,5:F2} snr: {5,5:F1} dB eoo: {6:F3}\n",
                _s, _i, st, sigDet ? 1 : 0, FreqOffset, SnrEstDb, _eooSmooth));
    }
}
