// Real-time factor of each stage on the fixture speech (9.8 s). RTF < 1 means faster than real time.
using System.Diagnostics;
using System.Runtime.InteropServices;
using RadeSharp;
using RadeSharp.Vocoder;

RadeLog.Writer = null;
string fx = args.Length > 0 ? args[0] : "tests/fixtures";
var speech = MemoryMarshal.Cast<byte, short>(File.ReadAllBytes(Path.Combine(fx, "speech.s16"))).ToArray();
var features = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(fx, "features.f32"))).ToArray();
double seconds = speech.Length / 16000.0;

void Report(string name, Action a)
{
    a(); // warm-up / JIT
    var sw = Stopwatch.StartNew();
    a();
    Console.WriteLine($"{name,-28} {sw.Elapsed.TotalSeconds,7:F3} s  RTF {sw.Elapsed.TotalSeconds / seconds:F3}");
}

Report("LPCNet features", () =>
{
    var enc = new LpcnetEncoder();
    var f = new float[36];
    for (int i = 0; i + 160 <= speech.Length; i += 160) enc.ComputeFeatures(speech.AsSpan(i, 160), f);
});
Report("FARGAN synthesis", () =>
{
    var fs = new FarganStream();
    var pcm = new short[160];
    for (int i = 0; i + 36 <= features.Length; i += 36) fs.Push(features.AsSpan(i, 36), pcm);
});
foreach (var mode in new[] { RadeMode.V1, RadeMode.V2 })
{
    RadeComp[] iq = [];
    Report($"{mode} tx", () =>
    {
        using var m = new RadeModem(mode);
        var list = new List<RadeComp>();
        var buf = new RadeComp[m.TxEooSamples + m.TxSamplesPerFrame];
        for (int i = 0; i + m.FeaturesPerFrame <= features.Length; i += m.FeaturesPerFrame)
            list.AddRange(buf.AsSpan(0, m.Transmit(features.AsSpan(i, m.FeaturesPerFrame), buf)));
        iq = list.ToArray();
    });
    Report($"{mode} rx (synced)", () =>
    {
        using var m = new RadeModem(mode);
        var f = new float[m.FeaturesPerFrame];
        for (int pos = 0; pos + m.RxSamplesNeeded <= iq.Length;) { int n = m.RxSamplesNeeded; m.Receive(iq.AsSpan(pos, n), f); pos += n; }
    });
    var noise = new RadeComp[iq.Length];
    var rnd = new Random(1);
    for (int i = 0; i < noise.Length; i++) noise[i] = new((float)(rnd.NextDouble() - 0.5) * 0.1f, (float)(rnd.NextDouble() - 0.5) * 0.1f);
    Report($"{mode} rx (noise, searching)", () =>
    {
        using var m = new RadeModem(mode);
        var f = new float[m.FeaturesPerFrame];
        for (int pos = 0; pos + m.RxSamplesNeeded <= noise.Length;) { int n = m.RxSamplesNeeded; m.Receive(noise.AsSpan(pos, n), f); pos += n; }
    });
}
