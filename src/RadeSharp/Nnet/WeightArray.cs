// SPDX-License-Identifier: BSD-3-Clause
// Port of Opus dnn/nnet.h + dnn/parse_lpcnet_weights.c (Xiph.Org, Mozilla, Amazon).

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace RadeSharp.Nnet;

/// <summary>Weight element type, as <c>WEIGHT_TYPE_*</c> in nnet.h.</summary>
public enum WeightType
{
    Float = 0,
    Int = 1,
    QWeight = 2,
    Int8 = 3,
}

/// <summary>One named weight table (<c>WeightArray</c>). <see cref="Size"/> is in bytes, like the C field.</summary>
public sealed class WeightArray
{
    public required string Name { get; init; }
    public required WeightType Type { get; init; }
    public required int Size { get; init; }
    public required byte[] Data { get; init; }

    // Typed views are converted once and shared by every layer bound to this array
    // (a benign race at worst converts twice).
    private Array? _typed;

    public float[] AsFloats() => (float[])(_typed ??= MemoryMarshal.Cast<byte, float>(Data.AsSpan(0, Size)).ToArray());
    public sbyte[] AsInt8() => (sbyte[])(_typed ??= MemoryMarshal.Cast<byte, sbyte>(Data.AsSpan(0, Size)).ToArray());
    public int[] AsInts() => (int[])(_typed ??= MemoryMarshal.Cast<byte, int>(Data.AsSpan(0, Size)).ToArray());
}

/// <summary>Parser for Opus "DNNw" weight blobs (<c>parse_weights</c>).</summary>
public static class WeightBlob
{
    public const int BlockSize = 64;       // WEIGHT_BLOCK_SIZE
    public const int BlobVersion = 0;      // WEIGHT_BLOB_VERSION
    private const int NameLength = 44;

    /// <summary>Parses a blob into its arrays, in file order. Throws on a malformed blob.</summary>
    public static IReadOnlyList<WeightArray> Parse(ReadOnlySpan<byte> data)
    {
        var list = new List<WeightArray>();
        while (data.Length > 0)
        {
            if (data.Length < BlockSize) throw new InvalidDataException("truncated weight header");
            // WeightHead: char head[4]; int version; int type; int size; int block_size; char name[44];
            int type = BinaryPrimitives.ReadInt32LittleEndian(data[8..]);
            int size = BinaryPrimitives.ReadInt32LittleEndian(data[12..]);
            int blockSize = BinaryPrimitives.ReadInt32LittleEndian(data[16..]);
            var name = data.Slice(20, NameLength);
            if (blockSize < size || blockSize > data.Length - BlockSize || name[NameLength - 1] != 0 || size <= 0)
                throw new InvalidDataException("malformed weight record");
            int nul = name.IndexOf((byte)0);
            list.Add(new WeightArray
            {
                Name = Encoding.ASCII.GetString(name[..nul]),
                Type = (WeightType)type,
                Size = size,
                Data = data.Slice(BlockSize, size).ToArray(),
            });
            data = data[(BlockSize + blockSize)..];
        }
        return list;
    }
}

/// <summary>Name lookup over a parsed blob (the C code scans a NULL-terminated array).</summary>
public sealed class WeightSet
{
    private readonly Dictionary<string, WeightArray> _byName = new(StringComparer.Ordinal);

    public WeightSet(IEnumerable<WeightArray> arrays)
    {
        // find_array_entry() returns the first match, so keep the first occurrence.
        foreach (var a in arrays) _byName.TryAdd(a.Name, a);
    }

    public static WeightSet FromBlob(ReadOnlySpan<byte> blob) => new(WeightBlob.Parse(blob));

    public WeightArray? Find(string name) => _byName.GetValueOrDefault(name);
}
