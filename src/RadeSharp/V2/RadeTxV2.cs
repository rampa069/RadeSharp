// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_tx_v2.c (Copyright (C) 2025 David Rowe): RADE V2 transmitter.

using RadeSharp.Core;
using RadeSharp.Dsp;
using RadeSharp.Models;
using static RadeSharp.Dsp.RadeDsp;
using static RadeSharp.V2.OfdmV2;

namespace RadeSharp.V2;

internal sealed class RadeTxV2
{
    public const int FramesPerStepV2 = 4;
    public const int NumFeaturesV2 = 21;
    public const int LatentDimV2 = 56;

    private readonly RadeEncV2Model _encModel = BuiltinWeights.RadeEncV2.Value;
    private readonly RadeEncoderV2 _encState = new();
    public readonly OfdmV2 Ofdm = new();
    private float _dataSymbol = -1.0f;

    // SSB BPF, continuous across Process() and Eoo() like radae_v2.py's RADEv2Transmitter.
    private readonly Bpf? _bpf;

    /// <summary><c>rade_tx_v2_init</c>.</summary>
    public RadeTxV2(bool bpfEn)
    {
        if (bpfEn)
        {
            float bandwidth = 2700.0f - 300.0f;
            float centre = (2700.0f + 300.0f) / 2.0f;
            _bpf = new Bpf(BpfNtap, Fs, bandwidth, centre, Math.Max(NeooV2, NmfV2));
        }
    }

    public static int NFeaturesIn => FramesPerStepV2 * NbTotalFeaturesV2;
    public static int NSamplesOut => NmfV2;
    public static int NEooOut => NeooV2;

    /// <summary><c>rade_tx_v2_set_data_symbol</c>.</summary>
    public void SetDataSymbol(float symbol) => _dataSymbol = symbol;

    /// <summary><c>rade_tx_v2_process</c>.</summary>
    public int Process(Span<RadeComp> txOut, ReadOnlySpan<float> featuresIn)
    {
        Span<float> encFeatures = stackalloc float[FramesPerStepV2 * NumFeaturesV2];
        for (int i = 0; i < FramesPerStepV2; i++)
        {
            var src = featuresIn.Slice(i * NbTotalFeaturesV2, NbTotalFeaturesV2);
            var dst = encFeatures.Slice(i * NumFeaturesV2, NumFeaturesV2);
            for (int j = 0; j < NumUsedFeaturesV2; j++) dst[j] = src[j];
            dst[NumUsedFeaturesV2] = _dataSymbol;   // BPSK data symbol
        }
        Span<float> z = stackalloc float[LatentDimV2];
        _encState.Run(_encModel, z, encFeatures);
        int nOut = Ofdm.ModFrame(txOut, z);
        _bpf?.Process(txOut, txOut, nOut);
        return nOut;
    }

    /// <summary><c>rade_tx_v2_eoo</c>.</summary>
    public int Eoo(Span<RadeComp> txOut)
    {
        Ofdm.Eoo.AsSpan().CopyTo(txOut);
        _bpf?.Process(txOut, txOut, NeooV2);
        return NeooV2;
    }
}
