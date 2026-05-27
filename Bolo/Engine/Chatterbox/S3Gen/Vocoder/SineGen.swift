// Bolo/Engine/Chatterbox/S3Gen/Vocoder/SineGen.swift
import Foundation
import MLX
import MLXNN
import MLXRandom

/// Sine waveform generator for source excitation.
///
/// Mirrors `SineGen` in
/// `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.hifigan`. Generates
/// `harmonic_num + 1` sinusoidal channels at integer multiples of the
/// fundamental F0, then mixes voiced sine with unvoiced noise.
///
/// This module has **no learnable parameters**. The random uniform phases
/// and Gaussian noise are produced at forward time. For deterministic
/// parity tests the caller can pass pre-computed `randomPhases` and
/// `noise` overrides via `callAsFunction(_:randomPhases:noise:)`.
final class SineGen: Module {

    let sineAmp: Float
    let noiseStd: Float
    let harmonicNum: Int
    let samplingRate: Int
    let voicedThreshold: Float

    init(
        sampRate: Int,
        harmonicNum: Int = 0,
        sineAmp: Float = 0.1,
        noiseStd: Float = 0.003,
        voicedThreshold: Float = 0
    ) {
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.harmonicNum = harmonicNum
        self.samplingRate = sampRate
        self.voicedThreshold = voicedThreshold
        super.init()
    }

    /// Generate sine waveform from F0.
    ///
    /// - Parameters:
    ///   - f0: `(B, 1, T)` fundamental frequency in Hz.
    ///   - randomPhases: optional pre-computed `(B, harmonicNum, 1)` uniform
    ///     phases in `[-π, π]`. If `nil`, sampled fresh.
    ///   - noise: optional pre-computed `(B, harmonicNum+1, T)` standard
    ///     normal samples. If `nil`, sampled fresh.
    /// - Returns: tuple `(sineWaves, uv, noise)` of shape `(B, H+1, T)` each.
    func callAsFunction(
        _ f0: MLXArray,
        randomPhases: MLXArray? = nil,
        noise: MLXArray? = nil
    ) -> (sineWaves: MLXArray, uv: MLXArray, noise: MLXArray) {
        let B = f0.shape[0]
        let T = f0.shape[2]

        // Harmonic multipliers: (1, H+1, 1), i.e. [1, 2, …, H+1].
        let harmonics = MLXArray(Int32(1)..<Int32(harmonicNum + 2))
            .asType(.float32)
            .reshaped([1, harmonicNum + 1, 1])
        let fMat = f0 * harmonics / Float(samplingRate)           // (B, H+1, T)

        // Phase accumulation. cumsum along time then mod 2π.
        let twoPi: Float = 2 * .pi
        var thetaMat = twoPi * fMat.cumsum(axis: -1)
        thetaMat = MLX.remainder(thetaMat, MLXArray(twoPi))

        // Random per-harmonic phase: zero for fundamental, uniform for harmonics.
        let providedPhases: MLXArray
        if let p = randomPhases {
            providedPhases = p
        } else if harmonicNum > 0 {
            providedPhases = MLXRandom.uniform(
                low: -Float.pi, high: Float.pi, [B, harmonicNum, 1]
            )
        } else {
            providedPhases = MLXArray.zeros([B, 0, 1])
        }
        let zeroPhase = MLXArray.zeros([B, 1, 1])
        let phaseVec: MLXArray
        if harmonicNum > 0 {
            phaseVec = concatenated([zeroPhase, providedPhases], axis: 1)
        } else {
            phaseVec = zeroPhase
        }

        // sine_waves
        var sineWaves = sineAmp * MLX.sin(thetaMat + phaseVec)    // (B, H+1, T)

        // Voiced/unvoiced mask
        let uv = (f0 .> voicedThreshold).asType(.float32)         // (B, 1, T)

        // Additive noise component
        let noiseSamples = noise ?? MLXRandom.normal(sineWaves.shape)
        let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
        let noiseSignal = noiseAmp * noiseSamples

        // Combine
        sineWaves = sineWaves * uv + noiseSignal

        return (sineWaves, uv, noiseSignal)
    }
}

/// Source module for harmonic-plus-noise synthesis.
///
/// Mirrors `SourceModule` in the Python reference. Wraps `SineGen` and a
/// `Linear(harmonic_num + 1 → 1)` merge layer that produces the final source
/// excitation signal (post-`tanh`).
///
/// Weight key layout:
///   l_sin_gen      — no parameters
///   l_linear.{weight,bias}
final class SourceModule: Module {

    @ModuleInfo(key: "l_sin_gen") var lSinGen: SineGen
    @ModuleInfo(key: "l_linear") var lLinear: Linear

    let sineAmp: Float

    init(
        samplingRate: Int,
        harmonicNum: Int = 8,
        sineAmp: Float = 0.1,
        addNoiseStd: Float = 0.003,
        voicedThreshold: Float = 10
    ) {
        self.sineAmp = sineAmp
        self._lSinGen.wrappedValue = SineGen(
            sampRate: samplingRate,
            harmonicNum: harmonicNum,
            sineAmp: sineAmp,
            noiseStd: addNoiseStd,
            voicedThreshold: voicedThreshold
        )
        self._lLinear.wrappedValue = Linear(harmonicNum + 1, 1, bias: true)
        super.init()
    }

    /// - Parameter f0: `(B, T, 1)` F0 input.
    /// - Returns: `(sineMerge, noise, uv)` where `sineMerge` is `(B, T, 1)`.
    ///   `noise` and `uv` shapes match `(B, T, 1)`. `noise` is produced for
    ///   API parity but not consumed downstream.
    func callAsFunction(
        _ f0: MLXArray,
        sineGenRandomPhases: MLXArray? = nil,
        sineGenNoise: MLXArray? = nil
    ) -> (sineMerge: MLXArray, noise: MLXArray, uv: MLXArray) {
        // f0: (B, T, 1) -> (B, 1, T)
        let f0BCT = f0.transposed(0, 2, 1)
        let (sineWavs, uv, _) = lSinGen(
            f0BCT,
            randomPhases: sineGenRandomPhases,
            noise: sineGenNoise
        )
        // (B, H+1, T) -> (B, T, H+1)
        let sineWavsBTC = sineWavs.transposed(0, 2, 1)
        let uvBTC = uv.transposed(0, 2, 1)

        let sineMerge = MLX.tanh(lLinear(sineWavsBTC))           // (B, T, 1)

        // Discarded by the production pipeline, but produce for API parity.
        let noise = MLXRandom.normal(uvBTC.shape) * sineAmp / 3
        return (sineMerge, noise, uvBTC)
    }
}
