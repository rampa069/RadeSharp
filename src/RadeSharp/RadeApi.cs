// SPDX-License-Identifier: BSD-2-Clause
// Port of rade_c rade_api.h / rade_api.c (Copyright (C) 2024 David Rowe).
//
// One-to-one surface of rade_api.h: same names, constants, flags, argument
// order and return values, with pointers mapped to spans. See RadeAPIUse.md in
// rade_c for the semantics; RadeModem is the idiomatic wrapper over this.

#pragma warning disable IDE1006 // C API names kept on purpose

using System.Globalization;
using RadeSharp.V1;
using RadeSharp.V2;

namespace RadeSharp;

/// <summary>The <c>struct rade</c> context. Treat as an opaque handle.</summary>
public sealed class Rade
{
    internal int flags;
    internal int auxdata;
    internal int bottleneck;
    internal RadeTxV1? tx;
    internal RadeRxV1? rx;
    internal RadeTxV2? tx_v2;
    internal RadeRxV2? rx_v2;
    internal bool closed;

    internal bool IsV2 => (flags & RadeApi.RADE_MODE_V2) != 0;

    internal Rade() { }
}

/// <summary>V2 per-symbol receiver diagnostics (<c>struct rade_stats</c>). All zero for V1.</summary>
public struct RadeStats
{
    public int sync;           // 0 = idle, 1 = sync
    public float delta_hat;    // IIR-smoothed timing offset (samples)
    public float delta_hat_g;  // instantaneous timing offset (samples)
    public float freq_offset;  // IIR-smoothed frequency offset (Hz)
    public float gain;         // AGC gain applied to the current symbol
    public float snr_est;      // SNR estimate (dB)
}

/// <summary>The <c>rade_api.h</c> functions.</summary>
public static class RadeApi
{
    private const int VERSION = 2;        // bump on breaking API changes
    private const int VERSION_MINOR = 2;  // bump on non-breaking additions

    public const int RADE_MODEM_SAMPLE_RATE = 8000;
    public const int RADE_SPEECH_SAMPLE_RATE = 16000;

    /// <summary>int16 &lt;-&gt; float scale: nominal 1.0 maps to 16384 (6 dB headroom). Real-valued RX input uses 2/scale.</summary>
    public const float RADE_INT16_SCALE = 16384.0f;

    public const int RADE_USE_C_ENCODER = 0x1;
    public const int RADE_USE_C_DECODER = 0x2;
    public const int RADE_FOFF_TEST = 0x4;
    public const int RADE_VERBOSE_0 = 0x8;
    public const int RADE_MODE_V2 = 0x10;
    public const int RADE_VERBOSE_TERSE = 0x20;
    public const int RADE_VERBOSE_FULL = 0x40;
    public const int RADE_NO_TX_BPF = 0x80;

    /// <summary>Must be called before any other RADE function (no-op, kept for API parity).</summary>
    public static void rade_initialize() { }

    /// <summary>Call when done with RADE (no-op, kept for API parity).</summary>
    public static void rade_finalize() { }

    /// <summary>Creates a context. <paramref name="model_file"/> is ignored (built-in weights).</summary>
    public static Rade? rade_open(string model_file, int flags)
    {
        var r = new Rade { flags = flags, auxdata = 1, bottleneck = 3 };
        var ci = CultureInfo.InvariantCulture;
        RadeLog.Write($"rade_open: model_file={model_file} (ignored, using built-in weights)\n");
        if ((flags & RADE_MODE_V2) != 0)
        {
            r.tx_v2 = new RadeTxV2(bpfEn: (flags & RADE_NO_TX_BPF) == 0);
            r.rx_v2 = new RadeRxV2(bpfEn: true);
            if ((flags & RADE_VERBOSE_0) != 0) r.rx_v2.Verbose = 0;
            else if ((flags & RADE_VERBOSE_FULL) != 0) r.rx_v2.Verbose = 3;
            else if ((flags & RADE_VERBOSE_TERSE) != 0) r.rx_v2.Verbose = 2;
            else r.rx_v2.Verbose = 0;
            RadeLog.Write(string.Format(ci, "rade_open: V2 n_features_in={0} Nmf={1} Neoo={2}\n",
                RadeTxV2.NFeaturesIn, RadeTxV2.NSamplesOut, RadeTxV2.NEooOut));
        }
        else
        {
            r.tx = new RadeTxV1(r.bottleneck, r.auxdata != 0, bpfEn: false);
            r.rx = new RadeRxV1(r.bottleneck, r.auxdata != 0, bpfEn: true);
            if ((flags & RADE_VERBOSE_0) != 0) r.rx.Verbose = 0;
            RadeLog.Write(string.Format(ci, "rade_open: V1 n_features_in={0} Nmf={1} Neoo={2} n_eoo_bits={3}\n",
                RadeTxV1.NFeaturesIn, RadeTxV1.NSamplesOut, RadeTxV1.NEooOut, r.tx.NEooBits));
        }
        return r;
    }

    public static void rade_close(Rade r)
    {
        r.closed = true;
        r.tx = null; r.rx = null; r.tx_v2 = null; r.rx_v2 = null;
    }

    public static int rade_version() => VERSION;
    public static int rade_version_minor() => VERSION_MINOR;

    public static int rade_n_tx_out(Rade r) => Ck(r).IsV2 ? RadeTxV2.NSamplesOut : RadeTxV1.NSamplesOut;
    public static int rade_n_tx_eoo_out(Rade r) => Ck(r).IsV2 ? RadeTxV2.NEooOut : RadeTxV1.NEooOut;
    public static int rade_nin_max(Rade r) => Ck(r).IsV2 ? RadeRxV2.NinMax : RadeRxV1.NinMax;
    public static int rade_nin(Rade r) => Ck(r).IsV2 ? r.rx_v2!.Nin : r.rx!.Nin;
    public static int rade_n_features_in_out(Rade r) => Ck(r).IsV2 ? RadeTxV2.NFeaturesIn : RadeTxV1.NFeaturesIn;
    public static int rade_n_eoo_bits(Rade r) => Ck(r).IsV2 ? 0 : r.tx!.NEooBits;

    /// <summary>V1 only: EOO bits in +/-1 float form (not 1/0).</summary>
    public static void rade_tx_set_eoo_bits(Rade r, ReadOnlySpan<float> eoo_bits)
    {
        Ck(r);
        // rade_api.c dereferences r->tx unconditionally; in V2 the call has no effect.
        r.tx?.SetEooBits(eoo_bits);
    }

    /// <summary>Returns the number of samples written to <paramref name="tx_out"/>.</summary>
    public static int rade_tx(Rade r, Span<RadeComp> tx_out, ReadOnlySpan<float> features_in)
        => Ck(r).IsV2 ? r.tx_v2!.Process(tx_out, features_in) : r.tx!.Process(tx_out, features_in);

    /// <summary>Final End-of-Over frame (V1 and V2); returns samples written.</summary>
    public static int rade_tx_eoo(Rade r, Span<RadeComp> tx_eoo_out)
        => Ck(r).IsV2 ? r.tx_v2!.Eoo(tx_eoo_out) : r.tx!.Eoo(tx_eoo_out);

    /// <summary>
    /// Processes <c>rade_nin(r)</c> samples. Returns the number of feature floats written (0 if none).
    /// <paramref name="has_eoo_out"/> is set on End-of-Over; for V1 <paramref name="eoo_out"/> then
    /// holds the EOO soft bits (pass an empty span for V2).
    /// </summary>
    public static int rade_rx(Rade r, Span<float> features_out, out int has_eoo_out, Span<float> eoo_out, ReadOnlySpan<RadeComp> rx_in)
    {
        if (Ck(r).IsV2)
        {
            int ret2 = r.rx_v2!.Process(features_out, rx_in);
            has_eoo_out = (ret2 & 0x2) != 0 ? 1 : 0;
            return (ret2 & 0x1) != 0 ? RadeRxV2.FeaturesOut : 0;
        }
        int ret = r.rx!.Process(features_out, eoo_out, rx_in);
        has_eoo_out = (ret & 0x2) != 0 ? 1 : 0;
        return (ret & 0x1) != 0 ? RadeRxV1.NFeaturesOut : 0;
    }

    public static int rade_sync(Rade r) => Ck(r).IsV2 ? (r.rx_v2!.State == RadeRxV2.StateSync ? 1 : 0) : (r.rx!.Sync ? 1 : 0);
    public static float rade_freq_offset(Rade r) => Ck(r).IsV2 ? r.rx_v2!.FreqOffset : r.rx!.FreqOffset;
    public static float rade_snrdB_3k_est(Rade r) => Ck(r).IsV2 ? r.rx_v2!.SnrEstDb : r.rx!.SnrdB3kEst;

    /// <summary>V2 per-symbol diagnostics; all zero for V1.</summary>
    public static void rade_get_stats(Rade r, out RadeStats stats)
    {
        stats = default;
        if (Ck(r).IsV2)
        {
            var rx = r.rx_v2!;
            stats.sync = rx.State == RadeRxV2.StateSync ? 1 : 0;
            stats.delta_hat = rx.DeltaHat;
            stats.delta_hat_g = rx.DeltaHatG;
            stats.freq_offset = rx.FreqOffset;
            stats.gain = rx.Gain;
            stats.snr_est = rx.SnrEstDb;
        }
    }

    /// <summary>Test mode: disable unsync after this many seconds (0 = disabled). V1 only.</summary>
    public static void rade_set_disable_unsync(Rade r, float seconds)
    {
        if (Ck(r).IsV2) return;
        r.rx!.DisableUnsync = seconds;
    }

    /// <summary>V2 only: BPSK data symbol (+1.0 or -1.0) for the next modem frame.</summary>
    public static void rade_tx_set_data_symbol(Rade r, float symbol)
    {
        if (Ck(r).IsV2) r.tx_v2!.SetDataSymbol(symbol);
    }

    /// <summary>V2 only: last received BPSK data symbol (soft decision).</summary>
    public static float rade_rx_get_data_symbol(Rade r) => Ck(r).IsV2 ? r.rx_v2!.DataSymbol : 0.0f;

    /// <summary>V2 only: enable/disable the input AGC (on by default).</summary>
    public static void rade_rx_set_agc(Rade r, int enable)
    {
        if (Ck(r).IsV2) r.rx_v2!.AgcEn = enable != 0;
    }

    private static Rade Ck(Rade r)
    {
        ArgumentNullException.ThrowIfNull(r);
        ObjectDisposedException.ThrowIf(r.closed, r);
        return r;
    }
}
