// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_rx.c (Copyright (C) 2024 David Rowe): RADE V1 receiver
// with the search/candidate/sync state machine.

using System.Globalization;
using RadeSharp.Core;
using RadeSharp.Dsp;
using RadeSharp.Models;
using static RadeSharp.Dsp.RadeDsp;

namespace RadeSharp.V1;

internal sealed class RadeRxV1
{
    public const int RxBufSize = 2 * Nmf + M + Ncp;

    private readonly Ofdm _ofdm;
    private readonly Bpf? _bpf;
    private readonly Acq _acq;
    private readonly RadeDecModel _decModel;
    private RadeDecoderV1 _decState = new();

    private readonly bool _auxdata;
    private readonly int _numFeatures;
    private readonly bool _coarseMag = true;
    private readonly int _timeOffset = -16;

    public int State = StateSearch;
    private int _validCount;
    private int _syncedCount;
    private int _uwErrors;
    private readonly int _nmfUnsync;
    private readonly int _syncedCountOneSec;

    private int _tmax;
    private int _tmaxCandidate;
    private float _fmax;
    private RadeComp _rxPhase = One;
    private int _nin = Nmf;
    private readonly RadeComp[] _rxBuf = new RadeComp[RxBufSize];
    private readonly RadeComp[] _rxFiltered = new RadeComp[Nmf + M];
    private float _snrdB3kEst;
    private int _mf = 1;

    public int Verbose = 2;
    public float DisableUnsync;

    /// <summary><c>rade_rx_init</c> (built-in weights).</summary>
    public RadeRxV1(int bottleneck, bool auxdata, bool bpfEn)
    {
        _auxdata = auxdata;
        _numFeatures = NumFeatures + (auxdata ? 1 : 0);
        _ofdm = new Ofdm(bottleneck);
        _acq = new Acq(_ofdm, AcqFrange, AcqFstep);
        _decModel = BuiltinWeights.RadeDec(_numFeatures * FramesPerStep);
        if (bpfEn)
        {
            float wMin = _ofdm.W[0];
            float wMax = _ofdm.W[Nc - 1];
            float bandwidth = (float)(1.2f * (wMax - wMin) * Fs / (2.0f * Math.PI));
            float centre = (float)((wMax + wMin) * Fs / (2.0f * Math.PI) / 2.0f);
            _bpf = new Bpf(BpfNtap, Fs, bandwidth, centre, Fs);
        }
        _nmfUnsync = (int)(Tunsync * Fs / Nmf);
        _syncedCountOneSec = Fs / Nmf;
    }

    /// <summary><c>rade_rx_reset</c>.</summary>
    public void Reset()
    {
        State = StateSearch;
        _nin = Nmf;
        _validCount = 0;
        _syncedCount = 0;
        _uwErrors = 0;
        _rxPhase = One;
        _snrdB3kEst = 0.0f;
        _decState = new RadeDecoderV1();
        _bpf?.Reset();
        Array.Clear(_rxBuf);
    }

    public int Nin => _nin;
    public static int NinMax => Nmf + M;
    public static int NFeaturesOut => Nzmf * FramesPerStep * NbTotalFeatures;
    public static int NEooBits => (Ns - 1) * Nc * 2;
    public bool Sync => State == StateSync;
    public float SnrdB3kEst => _snrdB3kEst;
    public float FreqOffset => _fmax;

    /// <summary><c>rade_rx_sum_uw_errors</c>.</summary>
    public void SumUwErrors(int n) => _uwErrors += n;

    /// <summary><c>rade_rx_process</c>: returns 0x1 when features are valid, 0x2 on End-of-Over.</summary>
    public int Process(Span<float> featuresOut, Span<float> eooOut, ReadOnlySpan<RadeComp> rxIn)
    {
        float fs = Fs;
        int prevState = State;
        bool validOutput = false, endOfOver = false, uwFail = false;

        ReadOnlySpan<RadeComp> rxSamples = rxIn;
        if (_bpf != null)
        {
            _bpf.Process(_rxFiltered, rxIn, _nin);
            rxSamples = _rxFiltered;
        }

        if (_nin > 0)
        {
            Array.Copy(_rxBuf, _nin, _rxBuf, 0, RxBufSize - _nin);
            rxSamples[.._nin].CopyTo(_rxBuf.AsSpan(RxBufSize - _nin));
        }

        bool candidate;
        if (State == StateSearch || State == StateCandidate)
        {
            candidate = _acq.DetectPilots(_rxBuf, out _tmax, out _fmax);
        }
        else
        {
            float ffineStart = _fmax - 1.0f;
            float ffineEnd = _fmax + 1.0f;
            int tfineStart = _tmax > 8 ? _tmax - 8 : 0;
            int tfineEnd = _tmax + 8;
            float fmaxHat = _fmax;
            _acq.Refine(_rxBuf, ref _tmax, ref fmaxHat, tfineStart, tfineEnd, ffineStart, ffineEnd, 0.1f);
            _fmax = 0.9f * _fmax + 0.1f * fmaxHat;

            _acq.CheckPilots(_rxBuf, _tmax, _fmax, out candidate, out endOfOver);

            // Timing slips
            _nin = Nmf;
            if (_tmax >= Nmf - M)
            {
                _nin = Nmf + M;
                _tmax -= M;
            }
            if (_tmax < M)
            {
                _nin = Nmf - M;
                _tmax += M;
            }

            _syncedCount++;
            if (_syncedCount % _syncedCountOneSec == 0)
            {
                if (_uwErrors > UwErrorThresh) uwFail = true;
                _uwErrors = 0;
            }

            // Frequency offset correction
            float w = (float)(2.0f * Math.PI * _fmax / fs);
            Span<RadeComp> rxCorrected = stackalloc RadeComp[Nmf + M + Ncp];
            RadeComp rxPhase = default;
            for (int n = 0; n < Nmf + M + Ncp; n++)
            {
                rxPhase = Cmul(_rxPhase, Cexp(-w * (n + 1)));
                rxCorrected[n] = Cmul(_rxBuf[_tmax - Ncp + n], rxPhase);
            }
            _rxPhase = rxPhase;
            float phaseMag = Cabs(_rxPhase);
            _rxPhase = Cscale(_rxPhase, 1.0f / phaseMag);

            Span<float> zHat = stackalloc float[Nzmf * LatentDim];
            float snrEst = 0.0f;
            _ofdm.DemodFrame(zHat, rxCorrected, _timeOffset, endOfOver, _coarseMag, ref snrEst);
            _snrdB3kEst = 0.9f * _snrdB3kEst + 0.1f * snrEst;

            validOutput = !endOfOver;
            if (validOutput)
            {
                featuresOut[..NFeaturesOut].Clear();
                int uwErrorsTotal = 0;
                Span<float> decFeatures = stackalloc float[FramesPerStep * NumFeaturesAux];
                for (int c = 0; c < Nzmf; c++)
                {
                    _decState.Run(_decModel, decFeatures[..(FramesPerStep * _numFeatures)], zHat.Slice(c * LatentDim, LatentDim));
                    for (int i = 0; i < FramesPerStep; i++)
                    {
                        int outIdx = (c * FramesPerStep + i) * NbTotalFeatures;
                        for (int j = 0; j < NumFeatures; j++) featuresOut[outIdx + j] = decFeatures[i * _numFeatures + j];
                    }
                    // Unique word: first aux symbol of each group (they repeat)
                    if (_auxdata && decFeatures[NumFeatures] > 0) uwErrorsTotal++;
                }
                if (_auxdata) _uwErrors += uwErrorsTotal;
            }

            if (endOfOver)
            {
                Span<float> zHatEoo = stackalloc float[(Ns - 1) * Nc * 2];
                _ofdm.DemodEoo(zHatEoo, rxCorrected, _timeOffset);
                zHatEoo[..NEooBits].CopyTo(eooOut);
            }
        }

        if (Verbose == 2 || (Verbose == 1 && (State == StateSearch || State == StateCandidate || prevState == StateCandidate)))
            LogStatus(candidate, endOfOver);

        int nextState = State;
        if (State == StateSearch)
        {
            if (candidate)
            {
                nextState = StateCandidate;
                _tmaxCandidate = _tmax;
                _validCount = 1;
            }
        }
        else if (State == StateCandidate)
        {
            // Look for 3 consecutive matches with similar timing
            if (candidate && Math.Abs(_tmax - _tmaxCandidate) < Ncp)
            {
                _validCount++;
                if (_validCount > 3)
                {
                    nextState = StateSync;
                    _decState = new RadeDecoderV1();
                    _syncedCount = 0;
                    _uwErrors = 0;
                    _validCount = _nmfUnsync;

                    float ffineStart = _fmax - 10.0f;
                    float ffineEnd = _fmax + 10.0f;
                    int tfineStart = _tmax > 1 ? _tmax - 1 : 0;
                    int tfineEnd = _tmax + 2;
                    _acq.Refine(_rxBuf, ref _tmax, ref _fmax, tfineStart, tfineEnd, ffineStart, ffineEnd, 0.25f);
                }
            }
            else nextState = StateSearch;
        }
        else if (State == StateSync)
        {
            bool unsyncEnable = true;
            if (DisableUnsync > 0.0f)
            {
                int disableAfterFrames = (int)(DisableUnsync * fs / Nmf);
                if (_syncedCount > disableAfterFrames) unsyncEnable = false;
            }
            if (candidate) _validCount = _nmfUnsync;
            else
            {
                _validCount--;
                if (unsyncEnable && _validCount == 0) nextState = StateSearch;
            }
            if (unsyncEnable && (endOfOver || uwFail)) nextState = StateSearch;
        }

        State = nextState;
        if (State == StateSearch) _nin = Nmf;
        _mf++;
        return (validOutput ? 0x1 : 0) | (endOfOver ? 0x2 : 0);
    }

    private void LogStatus(bool candidate, bool endOfOver)
    {
        if (RadeLog.Writer == null) return;
        var ci = CultureInfo.InvariantCulture;
        string stateStr = State == StateSearch ? "search" : State == StateCandidate ? "candidate" : "sync";
        var s = string.Format(ci, "{0,3} state: {1,10} valid: {2} {3} {4,2} Dthresh: {5,8:F2} Dtmax12: {6,8:F2} {7,8:F2} tmax: {8,4} fmax: {9,6:F2} SNRdB: {10,5:F2}",
            _mf, stateStr, candidate ? 1 : 0, endOfOver ? 1 : 0, _validCount, _acq.Dthresh, _acq.Dtmax12, _acq.Dtmax12Eoo, _tmax, _fmax, _snrdB3kEst);
        if (_auxdata && State == StateSync) s += " uw_err: " + _uwErrors.ToString(ci);
        RadeLog.Write(s + "\n");
    }
}
