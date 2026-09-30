// SPDX-License-Identifier: BSD-2-Clause
// Port of freedv-gui src/pipeline/rade_text.c (Mooneer Salem, 2024): reliable
// text for the RADE End-of-Over frame. Wire format: CRC8 (poly 0x1D) + up to 8
// 6-bit characters packed into 56 bits, LDPC HRA_56_56 -> 112 bits,
// Golden-Prime interleaved, mapped to 56 QPSK symbols (112 floats).
//
// This file is BSD-2 but links the LGPL-2.1 LDPC code in this assembly.

#pragma warning disable IDE1006 // C API names kept on purpose

namespace RadeSharp.Text;

/// <summary>Callback for a received, CRC-checked text (<c>on_text_rx_t</c>).</summary>
public delegate void RadeTextRxCallback(RadeText rt, string text, object? state);

/// <summary>The <c>rade_text_t</c> object.</summary>
public sealed class RadeText
{
    public const int LdpcTotalSizeBits = 112;
    public const int MaxLength = 8;
    private const int CrcLength = 1;
    private const int MaxRawLength = MaxLength + CrcLength;
    private const int BytesPerEncodedSegment = 8;

    private readonly LdpcCode _ldpc = LdpcCode.Hra56_56;
    private RadeTextRxCallback? _callback;
    private object? _callbackState;
    private readonly byte[] _txText = new byte[LdpcTotalSizeBits];
    private readonly RadeComp[] _inboundPendingSyms = new RadeComp[LdpcTotalSizeBits / 2];
    private readonly float[] _inboundPendingAmps = new float[LdpcTotalSizeBits / 2];
    private bool _enableStats;

    // Stats reference (process-global statics in the C code).
    private readonly float[] _lastEncodedLdpc = new float[LdpcTotalSizeBits];
    private readonly byte[] _lastLdpcAsBits = new byte[LdpcTotalSizeBits];
    private int _unusedEooBitCount, _unusedEooErrCount;

    /// <summary>Last BER statistics when stats are enabled (raw, coded).</summary>
    public (float RawBer, float CodedBer) LastStats { get; private set; }

    // ---- rade_text.h surface -------------------------------------------------

    public static RadeText rade_text_create() => new();
    public static void rade_text_destroy(RadeText ptr) { }

    public static void rade_text_generate_tx_string(RadeText ptr, string str, int strlength, Span<float> syms, int symSize)
        => ptr.GenerateTxString(str, strlength, syms, symSize);

    public static void rade_text_set_rx_callback(RadeText ptr, RadeTextRxCallback? fn, object? state)
    {
        ptr._callback = fn;
        ptr._callbackState = state;
    }

    public static void rade_text_rx(RadeText ptr, ReadOnlySpan<float> syms, int symSize) => ptr.Rx(syms, symSize);

    public static void rade_text_enable_stats_output(RadeText ptr, int enable) => ptr._enableStats = enable != 0;

    // ---- implementation -------------------------------------------------------

    // 6-bit charset: 0 null, 1-9 ASCII 38-47, 10-19 '0'-'9', 20-46 'A'-'Z', 47 ' '
    private static int ToOta(ReadOnlySpan<byte> input, Span<byte> output, int maxLength)
    {
        int o = 0;
        for (int i = 0; i < maxLength && i < input.Length; i++)
        {
            byte c = input[i];
            if (c == 0) break;
            if (c >= 38 && c <= 47) output[o++] = (byte)(c - 37);
            else if (c >= '0' && c <= '9') output[o++] = (byte)(c - '0' + 10);
            else if (c >= 'A' && c <= 'Z') output[o++] = (byte)(c - 'A' + 20);
            else if (c >= 'a' && c <= 'z') output[o++] = (byte)(char.ToUpperInvariant((char)c) - 'A' + 20);
        }
        output[o] = 0;
        return o;
    }

    private static int FromOta(ReadOnlySpan<byte> input, Span<byte> output, int maxLength)
    {
        int o = 0;
        for (int i = 0; i < maxLength; i++)
        {
            byte c = input[i];
            if (c == 0) break;
            if (c >= 1 && c <= 9) output[o++] = (byte)(c + 37);
            else if (c >= 10 && c <= 19) output[o++] = (byte)(c - 10 + '0');
            else if (c >= 20 && c <= 46) output[o++] = (byte)(c - 20 + 'A');
        }
        output[o] = 0;
        return o;
    }

    /// <summary><c>calculateCRC8_</c>: poly 0x1D, stops at the first NUL.</summary>
    internal static byte Crc8(ReadOnlySpan<byte> input, int length)
    {
        const byte generator = 0x1D;
        byte crc = 0;
        for (int p = 0; length > 0; p++)
        {
            byte ch = input[p];
            length--;
            if (ch == 0) break;
            crc ^= ch;
            for (int i = 0; i < 8; i++)
                crc = (crc & 0x80) != 0 ? (byte)((crc << 1) ^ generator) : (byte)(crc << 1);
        }
        return crc;
    }

    /// <summary>
    /// <c>rade_text_generate_tx_string</c>: fills <paramref name="syms"/> (V1: rade_n_eoo_bits floats)
    /// for <c>rade_tx_set_eoo_bits</c>. Floats past the 112 LDPC ones get the known 1,0,1,0 fill.
    /// </summary>
    public void GenerateTxString(string str, int strlength, Span<float> syms, int symSize)
    {
        Span<byte> tmp = stackalloc byte[MaxRawLength + 1];
        tmp.Clear();
        var ascii = System.Text.Encoding.Latin1.GetBytes(str);
        ToOta(ascii, tmp[CrcLength..], strlength < MaxLength ? strlength : MaxLength);
        int txtLength = tmp[CrcLength..].IndexOf((byte)0);
        if (txtLength >= MaxLength) txtLength = MaxLength;
        tmp[0] = Crc8(tmp[CrcLength..], txtLength);

        Span<byte> ibits = stackalloc byte[LdpcTotalSizeBits / 2];
        Span<byte> pbits = stackalloc byte[LdpcTotalSizeBits / 2];
        ibits.Clear();
        pbits.Clear();
        for (int i = 0; i < 8; i++) if ((tmp[0] & (1 << i)) != 0) ibits[i] = 1;
        for (int bi = 8; bi < LdpcTotalSizeBits / 2; bi++)
        {
            int bitsFromCrc = bi - 8;
            uint b = tmp[CrcLength + bitsFromCrc / 6];
            if ((b & (1u << (bitsFromCrc % 6))) != 0) ibits[bi] = 1;
        }
        Ldpc.Encode(_ldpc, ibits, pbits);

        Span<byte> tmpbits = stackalloc byte[LdpcTotalSizeBits];
        ibits.CopyTo(tmpbits);
        pbits.CopyTo(tmpbits[(LdpcTotalSizeBits / 2)..]);
        tmpbits.CopyTo(_lastLdpcAsBits);
        GpInterleaver.InterleaveBits(_txText, tmpbits, LdpcTotalSizeBits / 2);

        for (int i = 0; i < LdpcTotalSizeBits / 2; i++)
        {
            (syms[2 * i], syms[2 * i + 1]) = (_txText[2 * i], _txText[2 * i + 1]) switch
            {
                (0, 0) => (1f, 0f),
                (0, 1) => (0f, 1f),
                (1, 0) => (0f, -1f),
                _ => (-1f, 0f),
            };
        }
        if (_enableStats) syms[..LdpcTotalSizeBits].CopyTo(_lastEncodedLdpc);

        // Stuff the rest of the EOO with a known sequence (zeros would upset the decoder).
        for (int i = LdpcTotalSizeBits; i < symSize; i++) syms[i] = i % 2 != 0 ? 0 : 1;
    }

    /// <summary>
    /// <c>rade_text_rx</c>: decodes the V1 EOO soft bits. <paramref name="symSize"/> is the number of
    /// complex symbols (freedv-gui passes rade_n_eoo_bits()/2). Invokes the callback on a valid CRC.
    /// </summary>
    public void Rx(ReadOnlySpan<float> syms, int symSize)
    {
        var comp = System.Runtime.InteropServices.MemoryMarshal.Cast<float, RadeComp>(syms);
        GpInterleaver.DeinterleaveComp(_inboundPendingSyms, comp, LdpcTotalSizeBits / 2);

        float rms = 0;
        _unusedEooBitCount = 0;
        _unusedEooErrCount = 0;
        for (int i = 0; i < symSize; i++)
        {
            if (i < LdpcTotalSizeBits / 2)
            {
                var s = _inboundPendingSyms[i];
                rms += s.Real * s.Real + s.Imag * s.Imag;
            }
            else if (_enableStats && 2 * i + 1 < syms.Length)   // C reads past the payload here
            {
                _unusedEooBitCount += 2;
                if (syms[2 * i + 1] < 0) _unusedEooErrCount++;
            }
        }
        rms = MathF.Sqrt(rms / symSize);
        for (int i = 0; i < LdpcTotalSizeBits / 2; i++) _inboundPendingAmps[i] = rms;

        Span<byte> rawStr = stackalloc byte[MaxRawLength + 1];
        Span<byte> decodedStr = stackalloc byte[MaxRawLength + 1];
        rawStr.Clear();
        decodedStr.Clear();
        if (LdpcDecode(rawStr, rms))
        {
            int n = FromOta(rawStr[CrcLength..], decodedStr[CrcLength..], MaxLength);
            byte receivedCrc = rawStr[0];
            byte calcCrc = Crc8(rawStr[CrcLength..], MaxLength);
            if (receivedCrc == calcCrc && _callback != null)
                _callback(this, System.Text.Encoding.ASCII.GetString(decodedStr.Slice(CrcLength, n)), _callbackState);
        }
    }

    private bool LdpcDecode(Span<byte> dest, float meanAmplitude)
    {
        Span<float> llr = stackalloc float[LdpcTotalSizeBits];
        Span<byte> output = stackalloc byte[LdpcTotalSizeBits];
        int parityCheckCount = 0;

        int bitsRaw = 0, errorsRaw = 0;
        if (_enableStats)
        {
            var pending = System.Runtime.InteropServices.MemoryMarshal.Cast<RadeComp, float>(_inboundPendingSyms);
            for (int i = 0; i < LdpcTotalSizeBits; i++)
            {
                bitsRaw++;
                if (_lastEncodedLdpc[i] * pending[i] < 0) errorsRaw++;
            }
        }

        const float esNo = 3.0f;   // constant from freedv_700.c
        Ldpc.SymbolsToLlrs(llr, _inboundPendingSyms, _inboundPendingAmps, esNo, meanAmplitude, LdpcTotalSizeBits / 2);
        Ldpc.RunDecoder(_ldpc, output, llr, ref parityCheckCount);

        // Data is valid if BER < 0.2
        float berEst = (float)(_ldpc.NumberParityBits - parityCheckCount) / _ldpc.NumberParityBits;
        bool result = berEst < 0.2;

        if (_enableStats)
        {
            int bitsCoded = 0, errorsCoded = 0;
            for (int i = 0; i < LdpcTotalSizeBits / 2; i++)
            {
                bitsCoded++;
                if (_lastLdpcAsBits[i] != output[i]) errorsCoded++;
            }
            LastStats = ((float)(errorsRaw / (bitsRaw + 1E-12)), (float)(errorsCoded / (bitsCoded + 1E-12)));
        }

        if (result)
        {
            dest[..BytesPerEncodedSegment].Clear();
            for (int b = 0; b < 8; b++) if (output[b] != 0) dest[0] |= (byte)(1 << b);
            for (int b = 8; b < LdpcTotalSizeBits / 2; b++)
            {
                int bitsSinceCrc = b - 8;
                if (output[b] != 0) dest[1 + bitsSinceCrc / 6] |= (byte)(1 << (bitsSinceCrc % 6));
            }
        }
        return result;
    }
}
