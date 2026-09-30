// SPDX-License-Identifier: BSD-2-Clause

namespace RadeSharp;

/// <summary>
/// Destination of the diagnostic text the C library prints to stderr
/// (rade_open banners, per-frame receiver status). Defaults to
/// <see cref="Console.Error"/> like the C code; set to <c>null</c> to silence.
/// </summary>
public static class RadeLog
{
    public static TextWriter? Writer { get; set; } = Console.Error;

    internal static void Write(string s) => Writer?.Write(s);
}
