// SPDX-License-Identifier: BSD-3-Clause
// Port of Opus dnn/nnet.h LinearLayer / Conv2dLayer and linear_init()/conv2d_init().

namespace RadeSharp.Nnet;

/// <summary>Generic sparse affine transformation (<c>LinearLayer</c>).</summary>
public sealed class LinearLayer
{
    private const int SparseBlockSize = 32;

    public float[]? Bias;
    public float[]? Subias;
    public sbyte[]? Weights;
    public float[]? FloatWeights;
    public int[]? WeightsIdx;
    public float[]? Diag;
    public float[]? Scale;
    public int NbInputs;
    public int NbOutputs;

    /// <summary>
    /// <c>linear_init()</c>: binds the named arrays, validating sizes exactly like the C code.
    /// Throws <see cref="InvalidDataException"/> where the C function returns 1.
    /// </summary>
    public static LinearLayer Init(WeightSet arrays, string? bias, string? subias, string? weights,
        string? floatWeights, string? weightsIdx, string? diag, string? scale, int nbInputs, int nbOutputs)
    {
        var layer = new LinearLayer();
        if (bias != null) layer.Bias = FindCheck(arrays, bias, nbOutputs * sizeof(float)).AsFloats();
        if (subias != null) layer.Subias = FindCheck(arrays, subias, nbOutputs * sizeof(float)).AsFloats();
        if (weightsIdx != null)
        {
            layer.WeightsIdx = FindIdxCheck(arrays, weightsIdx, nbInputs, nbOutputs, out int totalBlocks);
            if (weights != null)
                layer.Weights = FindCheck(arrays, weights, SparseBlockSize * totalBlocks).AsInt8();
            if (floatWeights != null)
                layer.FloatWeights = OptCheck(arrays, floatWeights, SparseBlockSize * totalBlocks * sizeof(float))?.AsFloats();
        }
        else
        {
            if (weights != null)
                layer.Weights = FindCheck(arrays, weights, nbInputs * nbOutputs).AsInt8();
            if (floatWeights != null)
                layer.FloatWeights = OptCheck(arrays, floatWeights, nbInputs * nbOutputs * sizeof(float))?.AsFloats();
        }
        if (diag != null) layer.Diag = FindCheck(arrays, diag, nbOutputs * sizeof(float)).AsFloats();
        if (weights != null) layer.Scale = FindCheck(arrays, scale!, nbOutputs * sizeof(float)).AsFloats();
        layer.NbInputs = nbInputs;
        layer.NbOutputs = nbOutputs;
        return layer;
    }

    internal static WeightArray FindCheck(WeightSet arrays, string name, int size)
    {
        var a = arrays.Find(name);
        if (a == null || a.Size != size) throw new InvalidDataException($"weight array '{name}' missing or wrong size");
        return a;
    }

    internal static WeightArray? OptCheck(WeightSet arrays, string name, int size)
    {
        var a = arrays.Find(name);
        if (a == null) return null;
        if (a.Size != size) throw new InvalidDataException($"weight array '{name}' has wrong size");
        return a;
    }

    private static int[] FindIdxCheck(WeightSet arrays, string name, int nbIn, int nbOut, out int totalBlocks)
    {
        totalBlocks = 0;
        var a = arrays.Find(name) ?? throw new InvalidDataException($"weight index '{name}' missing");
        int[] idx = a.AsInts();
        int p = 0, remain = idx.Length;
        while (remain > 0)
        {
            int nbBlocks = idx[p++];
            if (remain < nbBlocks + 1) throw new InvalidDataException($"weight index '{name}' corrupt");
            for (int i = 0; i < nbBlocks; i++)
            {
                int pos = idx[p++];
                if (pos + 3 >= nbIn || (pos & 0x3) != 0) throw new InvalidDataException($"weight index '{name}' corrupt");
            }
            nbOut -= 8;
            remain -= nbBlocks + 1;
            totalBlocks += nbBlocks;
        }
        if (nbOut != 0) throw new InvalidDataException($"weight index '{name}' corrupt");
        return idx;
    }
}

/// <summary>2D convolution layer (<c>Conv2dLayer</c>).</summary>
public sealed class Conv2dLayer
{
    public float[]? Bias;
    public float[]? FloatWeights;
    public int InChannels;
    public int OutChannels;
    public int KTime;
    public int KHeight;

    /// <summary><c>conv2d_init()</c>.</summary>
    public static Conv2dLayer Init(WeightSet arrays, string? bias, string? floatWeights,
        int inChannels, int outChannels, int ktime, int kheight)
    {
        var layer = new Conv2dLayer();
        if (bias != null) layer.Bias = LinearLayer.FindCheck(arrays, bias, outChannels * sizeof(float)).AsFloats();
        if (floatWeights != null)
            layer.FloatWeights = LinearLayer.OptCheck(arrays, floatWeights,
                inChannels * outChannels * ktime * kheight * sizeof(float))?.AsFloats();
        layer.InChannels = inChannels;
        layer.OutChannels = outChannels;
        layer.KTime = ktime;
        layer.KHeight = kheight;
        return layer;
    }
}
