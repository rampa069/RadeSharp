// SPDX-License-Identifier: BSD-2-Clause

using static RadeSharp.RadeApi;

namespace RadeSharp;

/// <summary>RADE waveform version.</summary>
public enum RadeMode
{
    V1,
    /// <summary>Under active development upstream; not interoperable across versions.</summary>
    V2,
}

/// <summary>Result of one <see cref="RadeModem.Receive"/> call.</summary>
public readonly record struct RadeRxResult(int FeatureCount, bool EndOfOver)
{
    public bool HasFeatures => FeatureCount > 0;
}

/// <summary>
/// Idiomatic wrapper over the <see cref="RadeApi"/> context: one transmitter and
/// one receiver working on FARGAN feature vectors and 8 kHz complex IQ.
/// Not thread-safe; use one instance per stream.
/// </summary>
public sealed class RadeModem : IDisposable
{
    private readonly Rade _r;
    private float[]? _eooScratch;

    /// <param name="txBandpass">V2 only: SSB bandpass filter on the Tx output (default on).</param>
    public RadeModem(RadeMode mode = RadeMode.V1, bool txBandpass = true, bool verbose = false)
    {
        int flags = verbose ? RADE_VERBOSE_TERSE : RADE_VERBOSE_0;
        if (mode == RadeMode.V2) flags |= RADE_MODE_V2;
        if (!txBandpass) flags |= RADE_NO_TX_BPF;
        _r = rade_open("", flags) ?? throw new InvalidOperationException("rade_open failed");
        Mode = mode;
    }

    public RadeMode Mode { get; }

    /// <summary>The underlying 1:1 API handle, for calls not wrapped here.</summary>
    public Rade Handle => _r;

    public const int ModemSampleRate = RADE_MODEM_SAMPLE_RATE;
    public const int SpeechSampleRate = RADE_SPEECH_SAMPLE_RATE;

    /// <summary>Feature floats per <see cref="Transmit"/> input and per <see cref="Receive"/> output.</summary>
    public int FeaturesPerFrame => rade_n_features_in_out(_r);
    public int TxSamplesPerFrame => rade_n_tx_out(_r);
    public int TxEooSamples => rade_n_tx_eoo_out(_r);
    /// <summary>Upper bound of <see cref="RxSamplesNeeded"/>, for buffer allocation.</summary>
    public int RxMaxSamples => rade_nin_max(_r);
    /// <summary>Exact number of IQ samples the next <see cref="Receive"/> consumes; changes with timing tracking.</summary>
    public int RxSamplesNeeded => rade_nin(_r);
    /// <summary>V1 End-of-Over soft bits count (0 for V2).</summary>
    public int EooBitCount => rade_n_eoo_bits(_r);

    public bool InSync => rade_sync(_r) != 0;
    public float FrequencyOffsetHz => rade_freq_offset(_r);
    public float SnrDb3k => rade_snrdB_3k_est(_r);

    public RadeStats Stats
    {
        get
        {
            rade_get_stats(_r, out var s);
            return s;
        }
    }

    /// <summary>V2: receiver input AGC (on by default).</summary>
    public bool RxAgc { set => rade_rx_set_agc(_r, value ? 1 : 0); }

    /// <summary>V2: the soft BPSK aux-channel symbol of the last decoded frame.</summary>
    public float RxDataSymbol => rade_rx_get_data_symbol(_r);

    /// <summary>V2: BPSK aux-channel symbol (+1/-1) sent with the next frame.</summary>
    public void SetTxDataSymbol(float symbol) => rade_tx_set_data_symbol(_r, symbol);

    /// <summary>V1: +/-1 soft bits carried in the End-of-Over frame.</summary>
    public void SetEooBits(ReadOnlySpan<float> bits)
    {
        if (bits.Length < EooBitCount) throw new ArgumentException($"need {EooBitCount} bits", nameof(bits));
        rade_tx_set_eoo_bits(_r, bits);
    }

    /// <summary>Encodes one frame of features; returns IQ samples written.</summary>
    public int Transmit(ReadOnlySpan<float> features, Span<RadeComp> iqOut)
    {
        if (features.Length < FeaturesPerFrame) throw new ArgumentException($"need {FeaturesPerFrame} features", nameof(features));
        if (iqOut.Length < TxSamplesPerFrame) throw new ArgumentException($"need {TxSamplesPerFrame} samples", nameof(iqOut));
        return rade_tx(_r, iqOut, features);
    }

    /// <summary>Generates the End-of-Over frame; returns IQ samples written.</summary>
    public int TransmitEndOfOver(Span<RadeComp> iqOut)
    {
        if (iqOut.Length < TxEooSamples) throw new ArgumentException($"need {TxEooSamples} samples", nameof(iqOut));
        return rade_tx_eoo(_r, iqOut);
    }

    /// <summary>
    /// Consumes exactly <see cref="RxSamplesNeeded"/> samples from <paramref name="iqIn"/>.
    /// <paramref name="eooBits"/> (V1) receives the End-of-Over soft bits when <see cref="RadeRxResult.EndOfOver"/>.
    /// </summary>
    public RadeRxResult Receive(ReadOnlySpan<RadeComp> iqIn, Span<float> featuresOut, Span<float> eooBits = default)
    {
        int nin = RxSamplesNeeded;
        if (iqIn.Length < nin) throw new ArgumentException($"need {nin} samples", nameof(iqIn));
        if (featuresOut.Length < FeaturesPerFrame) throw new ArgumentException($"need {FeaturesPerFrame} floats", nameof(featuresOut));
        if (Mode == RadeMode.V1 && eooBits.Length < EooBitCount)
            eooBits = _eooScratch ??= new float[EooBitCount];
        int n = rade_rx(_r, featuresOut, out int hasEoo, eooBits, iqIn[..nin]);
        return new RadeRxResult(n, hasEoo != 0);
    }

    public void Dispose() => rade_close(_r);
}
