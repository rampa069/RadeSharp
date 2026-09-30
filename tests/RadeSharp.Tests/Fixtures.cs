using System.Runtime.InteropServices;

namespace RadeSharp.Tests;

/// <summary>Access to the golden vectors in tests/fixtures (regenerate with reference/make_golden.sh).</summary>
internal static class Fixtures
{
    public static readonly string Dir = Locate();

    private static string Locate()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            var candidate = Path.Combine(d.FullName, "tests", "fixtures");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("tests/fixtures not found");
    }

    public static byte[] Bytes(string name) => File.ReadAllBytes(Path.Combine(Dir, name));
    public static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(Bytes(name)).ToArray();
    public static short[] Shorts(string name) => MemoryMarshal.Cast<byte, short>(Bytes(name)).ToArray();
    public static RadeComp[] Iq(string name) => MemoryMarshal.Cast<byte, RadeComp>(Bytes(name)).ToArray();
    public static T[] Records<T>(string name) where T : unmanaged => MemoryMarshal.Cast<byte, T>(Bytes(name)).ToArray();
}

internal static class Exact
{
    /// <summary>Asserts bit-exact equality and reports the first mismatch with context.</summary>
    public static void Equal(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        int mismatches = 0, first = -1;
        double maxDiff = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                if (first < 0) first = i;
                mismatches++;
                maxDiff = Math.Max(maxDiff, Math.Abs((double)expected[i] - actual[i]));
            }
        }
        Assert.True(mismatches == 0,
            $"{what}: {mismatches}/{expected.Length} differ, first at {first} (expected {(first >= 0 ? expected[first] : 0):R}, got {(first >= 0 ? actual[first] : 0):R}), max |diff| {maxDiff:E3}");
    }
}
