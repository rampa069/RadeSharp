// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_tx.c (Copyright (C) 2024 David Rowe): RADE V1 transmitter.

using RadeSharp.Core;
using RadeSharp.Dsp;
using RadeSharp.Models;
using static RadeSharp.Dsp.RadeDsp;

namespace RadeSharp.V1;

internal sealed class RadeTxV1
{
    public readonly Ofdm Ofdm;
    private readonly Bpf? _bpf;
    private readonly RadeEncModel _encModel;
    private RadeEncoderV1 _encState = new();
    private readonly int _bottleneck;
    private readonly bool _auxdata;
    private readonly int _numFeatures;
    private readonly float[] _eooBits = new float[Nc * (Ns - 1) * 2];

    public int NEooBits { get; }

    /// <summary><c>rade_tx_init</c> (built-in weights).</summary>
    public RadeTxV1(int bottleneck, bool auxdata, bool bpfEn)
    {
        _bottleneck = bottleneck;
        _auxdata = auxdata;
        _numFeatures = NumFeatures + (auxdata ? 1 : 0);
        Ofdm = new Ofdm(bottleneck);
        _encModel = BuiltinWeights.RadeEnc(_numFeatures * FramesPerStep);
        if (bpfEn)
        {
            float wMin = Ofdm.W[0];
            float wMax = Ofdm.W[Nc - 1];
            float bandwidth = (float)(1.2f * (wMax - wMin) * Fs / (2.0f * Math.PI));
            float centre = (float)((wMax + wMin) * Fs / (2.0f * Math.PI) / 2.0f);
            _bpf = new Bpf(BpfNtap, Fs, bandwidth, centre, Fs);
        }
        NEooBits = (Ns - 1) * Nc * 2;
    }

    /// <summary><c>rade_tx_reset</c>.</summary>
    public void Reset()
    {
        _encState = new RadeEncoderV1();
        _bpf?.Reset();
    }

    public static int NFeaturesIn => Nzmf * FramesPerStep * NbTotalFeatures;
    public static int NSamplesOut => Nmf;
    public static int NEooOut => Neoo;

    /// <summary><c>rade_tx_state_set_eoo_bits</c>.</summary>
    public void SetEooBits(ReadOnlySpan<float> bits) => bits[..NEooBits].CopyTo(_eooBits);

    /// <summary><c>rade_tx_process</c>.</summary>
    public int Process(Span<RadeComp> txOut, ReadOnlySpan<float> featuresIn)
    {
        Span<float> z = stackalloc float[Nzmf * LatentDim];
        Span<float> encFeatures = stackalloc float[FramesPerStep * NumFeaturesAux];
        for (int c = 0; c < Nzmf; c++)
        {
            for (int i = 0; i < FramesPerStep; i++)
            {
                int inIdx = (c * FramesPerStep + i) * NbTotalFeatures;
                for (int j = 0; j < NumFeatures; j++) encFeatures[i * _numFeatures + j] = featuresIn[inIdx + j];
                if (_auxdata) encFeatures[i * _numFeatures + NumFeatures] = -1.0f;
            }
            _encState.Run(_encModel, z.Slice(c * LatentDim, LatentDim), encFeatures[..(FramesPerStep * _numFeatures)], _bottleneck);
        }
        int nOut = Ofdm.ModFrame(txOut, z);
        if (_bpf != null) FilterAndClip(txOut, nOut);
        return nOut;
    }

    /// <summary><c>rade_tx_state_eoo</c>: EOO frame with the EOO bits in the data slots.</summary>
    public int Eoo(Span<RadeComp> txOut)
    {
        int nEoo = Ofdm.NEoo;
        Ofdm.Eoo.AsSpan(0, nEoo).CopyTo(txOut);

        Span<RadeComp> freqSym = stackalloc RadeComp[Nc];
        Span<RadeComp> timeSym = stackalloc RadeComp[M];
        for (int d = 0; d < Ns - 1; d++)
        {
            int framePos = d + 2;
            for (int c = 0; c < Nc; c++)
            {
                int b = (d * Nc + c) * 2;
                freqSym[c] = new(_eooBits[b], _eooBits[b + 1]);
            }
            Ofdm.Idft(timeSym, freqSym);
            for (int c = 0; c < M; c++)
            {
                timeSym[c] = Cscale(timeSym[c], Ofdm.PilotGain);
                if (Ofdm.Bottleneck == 3) timeSym[c] = TanhLimit(timeSym[c]);
            }
            Ofdm.InsertCp(txOut[(framePos * (M + Ncp))..], timeSym);
        }
        if (_bpf != null) FilterAndClip(txOut, nEoo);
        return nEoo;
    }

    private void FilterAndClip(Span<RadeComp> txOut, int n)
    {
        Span<RadeComp> filtered = stackalloc RadeComp[n];
        _bpf!.Process(filtered, txOut, n);
        for (int i = 0; i < n; i++)
        {
            float mag = Cabs(filtered[i]);
            txOut[i] = mag > 1.0f ? Cscale(filtered[i], 1.0f / mag) : filtered[i];
        }
    }
}
