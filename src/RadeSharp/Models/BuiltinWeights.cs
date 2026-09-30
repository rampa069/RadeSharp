// SPDX-License-Identifier: BSD-2-Clause

using RadeSharp.Nnet;

namespace RadeSharp.Models;

/// <summary>
/// The compiled-in weights (the C build links them as static tables). Each blob
/// is parsed once per process and the models are shared read-only between contexts.
/// </summary>
internal static class BuiltinWeights
{
    private static readonly Lazy<WeightSet> RadeV1EncSet = Load("rade_v1_enc.bin");
    private static readonly Lazy<WeightSet> RadeV1DecSet = Load("rade_v1_dec.bin");

    public static readonly Lazy<RadeEncV2Model> RadeEncV2 = new(() => new RadeEncV2Model(Load("rade_v2_enc.bin").Value));
    public static readonly Lazy<RadeDecV2Model> RadeDecV2 = new(() => new RadeDecV2Model(Load("rade_v2_dec.bin").Value));
    public static readonly Lazy<RadeSyncModel> RadeSync = new(() => new RadeSyncModel(Load("rade_v2_sync.bin").Value));
    public static readonly Lazy<FarganModel> Fargan = new(() => new FarganModel(Load("fargan.bin").Value));
    public static readonly Lazy<PitchDnnModel> PitchDnn = new(() => new PitchDnnModel(Load("pitchdnn.bin").Value));

    /// <summary>V1 models take a dimension argument (<c>init_radeenc(..., input_dim)</c>).</summary>
    public static RadeEncModel RadeEnc(int inputDim) => new(RadeV1EncSet.Value, inputDim);
    public static RadeDecModel RadeDec(int outputDim) => new(RadeV1DecSet.Value, outputDim);

    private static Lazy<WeightSet> Load(string file) => new(() =>
    {
        using var s = typeof(BuiltinWeights).Assembly.GetManifestResourceStream("RadeSharp.Weights." + file)
            ?? throw new InvalidOperationException($"embedded weights '{file}' missing");
        var data = new byte[s.Length];
        s.ReadExactly(data);
        return WeightSet.FromBlob(data);
    });
}
