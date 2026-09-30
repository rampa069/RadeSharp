// SPDX-License-Identifier: BSD-2-Clause
// Minimal WAV I/O matching rade_tx_wav.c / rade_rx_wav.c.

using System.Text;

namespace RadeSharp.Tools;

internal sealed record WavInfo(int SampleRate, int Channels, int BitsPerSample, bool IsFloat, long DataOffset, uint DataSize);

internal static class Wav
{
    public static WavInfo? ReadHeader(BinaryReader r)
    {
        if (Encoding.ASCII.GetString(r.ReadBytes(4)) != "RIFF") return null;
        r.ReadUInt32();
        if (Encoding.ASCII.GetString(r.ReadBytes(4)) != "WAVE") return null;
        int sr = 0, nch = 0, bps = 0;
        bool isFloat = false;
        var s = r.BaseStream;
        while (s.Position + 8 <= s.Length)
        {
            string id = Encoding.ASCII.GetString(r.ReadBytes(4));
            uint size = r.ReadUInt32();
            if (id == "fmt ")
            {
                if (size < 16) return null;
                ushort fmt = r.ReadUInt16();
                nch = r.ReadUInt16();
                sr = (int)r.ReadUInt32();
                r.ReadUInt32();
                r.ReadUInt16();
                bps = r.ReadUInt16();
                isFloat = fmt == 3;
                if (size > 16) s.Seek(size - 16, SeekOrigin.Current);
            }
            else if (id == "data") return new WavInfo(sr, nch, bps, isFloat, s.Position, size);
            else s.Seek((size + 1) & ~1u, SeekOrigin.Current);
        }
        return null;
    }

    /// <summary><c>wav_read_mono_float</c>: 16-bit PCM, channels averaged, scaled by 1/32768.</summary>
    public static float[] ReadMonoFloat(BinaryReader r, WavInfo info)
    {
        long mono = info.DataSize / (2 * info.Channels);
        var buf = new float[mono];
        for (long i = 0; i < mono; i++)
        {
            float sum = 0.0f;
            for (int ch = 0; ch < info.Channels; ch++) sum += r.ReadInt16() / 32768.0f;
            buf[i] = sum / info.Channels;
        }
        return buf;
    }

    /// <summary>rade_rx_wav.c's reader: real-valued modem audio scaled by 2/RADE_INT16_SCALE (see rade_api.h).</summary>
    public static float[] ReadModemReal(BinaryReader r, WavInfo info)
    {
        long n = info.DataSize / 2;
        var buf = new float[n];
        for (long i = 0; i < n; i++) buf[i] = r.ReadInt16() * (2.0f / RadeApi.RADE_INT16_SCALE);
        return buf;
    }

    public static void WriteHeader(BinaryWriter w, int sampleRate, uint dataBytes)
    {
        w.Write("RIFF"u8); w.Write(36 + dataBytes);
        w.Write("WAVE"u8);
        w.Write("fmt "u8); w.Write(16u);
        w.Write((ushort)1); w.Write((ushort)1);
        w.Write((uint)sampleRate); w.Write((uint)sampleRate * 2);
        w.Write((ushort)2); w.Write((ushort)16);
        w.Write("data"u8); w.Write(dataBytes);
    }

    /// <summary>float -> int16 as the C tools: clamp to +/-32767 then floor(0.5 + v) in double.</summary>
    public static short ToInt16(float v)
    {
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32767.0f) v = -32767.0f;
        return (short)Math.Floor(0.5 + (double)v);
    }
}
