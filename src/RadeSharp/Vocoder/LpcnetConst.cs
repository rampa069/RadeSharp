// SPDX-License-Identifier: BSD-3-Clause
// Constants from Opus dnn/lpcnet.h, freq.h, pitchdnn.h, lpcnet_private.h.

namespace RadeSharp.Vocoder;

/// <summary>LPCNet/FARGAN framing constants.</summary>
public static class LpcnetConst
{
    public const int NbFeatures = 20;
    public const int NbTotalFeatures = 36;
    public const int LpcnetFrameSize = 160;
    public const int LpcOrder = 16;
    public const float Preemphasis = 0.85f;
    public const int FrameSize5ms = 2;
    public const int OverlapSize5ms = 2;
    public const int TrainingOffset5ms = 1;
    public const int WindowSize5ms = FrameSize5ms + OverlapSize5ms;
    public const int FrameSize = 80 * FrameSize5ms;           // 160
    public const int OverlapSize = 80 * OverlapSize5ms;       // 160
    public const int TrainingOffset = 80 * TrainingOffset5ms;
    public const int WindowSize = FrameSize + OverlapSize;    // 320
    public const int FreqSize = WindowSize / 2 + 1;           // 161
    public const int NbBands = 18;
    public const int NbBands1 = NbBands - 1;
    public const int PitchMinPeriod = 32;
    public const int PitchMaxPeriod = 256;
    public const int NbXcorrFeatures = PitchMaxPeriod - PitchMinPeriod;
    public const int PitchFrameSize = 320;
    public const int PitchBufSize = PitchMaxPeriod + PitchFrameSize;
    public const int PitchIfMaxFreq = 30;
    public const int PitchIfFeatures = 3 * PitchIfMaxFreq - 2;
}
