using System.Diagnostics;

namespace RadeSharp.Tests;

/// <summary>The WAV convenience tools must produce the same bytes as rade_c's rade_tx_wav / rade_rx_wav.</summary>
public class CliTests
{
    private static void Cli(params string[] args)
    {
        var psi = new ProcessStartInfo("dotnet") { RedirectStandardError = true, UseShellExecute = false };
        psi.ArgumentList.Add(Path.Combine(AppContext.BaseDirectory, "radesharp-cli.dll"));
        foreach (var a in args) psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        string err = p.StandardError.ReadToEnd();
        p.WaitForExit();
        Assert.True(p.ExitCode == 0, err);
    }

    [Theory]
    [InlineData("v1")]
    [InlineData("v2")]
    public void WavToolsMatchC(string v)
    {
        string dir = Directory.CreateTempSubdirectory("radesharp").FullName;
        string input = Path.Combine(Fixtures.Dir, "..", "..", "..", "rade_c", "input_sample.wav");
        if (!File.Exists(input)) input = Path.Combine(dir, "in.wav");
        if (!File.Exists(input))
        {
            // speech.s16 is input_sample.wav without its header
            using var w = new BinaryWriter(File.Create(input));
            var pcm = Fixtures.Bytes("speech.s16");
            w.Write("RIFF"u8); w.Write(36 + pcm.Length); w.Write("WAVE"u8);
            w.Write("fmt "u8); w.Write(16); w.Write((short)1); w.Write((short)1); w.Write(16000); w.Write(32000); w.Write((short)2); w.Write((short)16);
            w.Write("data"u8); w.Write(pcm.Length); w.Write(pcm);
        }
        string[] flag = v == "v2" ? ["--v2"] : [];
        string tx = Path.Combine(dir, "tx.wav"), rx = Path.Combine(dir, "rx.wav");
        Cli(["rade_tx_wav", .. flag, "-v", "0", input, tx]);
        Cli(["rade_rx_wav", .. flag, "-v", "0", Path.Combine(Fixtures.Dir, $"wav_tx_{v}.wav"), rx]);
        Assert.True(Fixtures.Bytes($"wav_tx_{v}.wav").AsSpan().SequenceEqual(File.ReadAllBytes(tx)), "rade_tx_wav output differs from C");
        Assert.True(Fixtures.Bytes($"wav_rx_{v}.wav").AsSpan().SequenceEqual(File.ReadAllBytes(rx)), "rade_rx_wav output differs from C");
        Directory.Delete(dir, true);
    }
}
