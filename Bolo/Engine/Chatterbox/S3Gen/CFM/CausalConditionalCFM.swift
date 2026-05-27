// Bolo/Engine/Chatterbox/S3Gen/CFM/CausalConditionalCFM.swift
import Foundation
import MLX
import MLXNN

/// Classifier-free-guidance flow-matching solver wrapping `ConditionalDecoder`.
///
/// Mirrors `CausalConditionalCFM` in
/// `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.flow_matching`.
///
/// For Chatterbox-Turbo the decoder is configured with `meanflow=True` and the
/// CFM is called with `n_cfm_timesteps = 2` (so `t_span = [0, 0.5, 1.0]`,
/// two Euler steps). The meanflow path uses `_basic_euler` (no CFG) — the
/// decoder mixes `t_emb` with `r_emb` via `time_embed_mixer` internally.
///
/// For non-meanflow base Chatterbox the path is `_solve_euler_cfg` with
/// classifier-free guidance (untested here but implemented for completeness).
///
/// Inputs to `callAsFunction`:
///   - `mu`:    `(B, 80, T)` — encoder mel conditioning (`h_proj.T`).
///   - `mask`:  `(B, 1, T)`  — float mask over valid time positions.
///   - `noise`: `(B, 80, T)` — pinned initial noise `z` (caller supplies; in
///     production it would be sampled via `MLXRandom.normal(...)`. The Swift
///     port externalises the noise so parity tests can replay the Python RNG
///     bit-exactly.)
///   - `spks`:  `(B, 80)`    — speaker embedding (optional).
///   - `cond`:  `(B, 80, T)` — extra conditioning (zeros except over prompt
///     prefix), optional.
///   - `noisedMels`: `(B, 80, T')` — optional pre-noised mel suffix used to
///     overwrite the trailing part of `noise` (Turbo continuation pattern).
///   - `nTimesteps`: integer; default 2 for meanflow (Turbo), 10 for base.
///
/// Output: `(B, 80, T)` `speech_feat` mel-like tensor at `t=1`.
/// NOT a Module subclass: the CFM solver owns no parameters of its own.
/// Its only Module-valued field is `estimator`, supplied externally and
/// already mounted in the parameter tree by the host model. Making this a
/// Module would duplicate the decoder under both `decoder.*` and
/// `cfm.estimator.*` paths.
final class CausalConditionalCFM {

    /// Velocity-field estimator. NOTE: this wraps an existing decoder; it's
    /// NOT re-registered as a child module (the decoder owns its own params).
    /// We keep a strong reference but do not declare it `@ModuleInfo`.
    let estimator: ConditionalDecoder

    /// Schedule type: "cosine" or "linear". Meanflow path always uses linear.
    let tScheduler: String
    /// Classifier-free guidance weight (only used in the non-meanflow path).
    let inferenceCfgRate: Float
    /// Sigma_min (unused in current solver but kept for parity with Python).
    let sigmaMin: Float

    init(
        estimator: ConditionalDecoder,
        tScheduler: String = "cosine",
        inferenceCfgRate: Float = 0.7,
        sigmaMin: Float = 1e-6
    ) {
        self.estimator = estimator
        self.tScheduler = tScheduler
        self.inferenceCfgRate = inferenceCfgRate
        self.sigmaMin = sigmaMin
    }

    /// Forward CFM ODE solve.
    ///
    /// - Parameters:
    ///   - noise: pre-sampled `(B, 80, T)` noise `z` matching `mu.shape`.
    ///   - mu: `(B, 80, T)` encoder mel.
    ///   - mask: `(B, 1, T)` float mask.
    ///   - nTimesteps: number of Euler steps (default 2 for meanflow Turbo).
    ///   - spks: optional `(B, 80)`.
    ///   - cond: optional `(B, 80, T)`.
    ///   - noisedMels: optional pre-noised suffix `(B, 80, T')`; concatenated
    ///     onto the leading prefix of `noise` if provided.
    ///   - meanflow: select meanflow path (basic Euler, no CFG) vs CFG path.
    /// - Returns: `(B, 80, T)` speech_feat at the end of the ODE.
    func callAsFunction(
        noise: MLXArray,
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int = 2,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil,
        noisedMels: MLXArray? = nil,
        meanflow: Bool = true
    ) -> MLXArray {
        var z = noise

        // If a pre-noised mel suffix is provided, splice it onto the prefix
        // of `z`. (Python: z = concat([z[:, :, :prompt_len], noised_mels], 2))
        if let noisedMels {
            let promptLen = mu.shape[2] - noisedMels.shape[2]
            if promptLen > 0 {
                z = concatenated(
                    [z[0..., 0..., 0..<promptLen], noisedMels],
                    axis: 2
                )
            } else {
                z = noisedMels
            }
        }

        // Time grid: linspace(0, 1, nTimesteps + 1). For non-meanflow + cosine
        // scheduler, apply 1 - cos(t * pi / 2).
        let n = nTimesteps + 1
        var tSpan = MLXArray.linspace(Float(0), Float(1), count: n)
        if !meanflow && tScheduler == "cosine" {
            tSpan = 1.0 - MLX.cos(tSpan * 0.5 * Float.pi)
        }

        if meanflow {
            return basicEuler(
                x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond
            )
        } else {
            return solveEulerCFG(
                x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond
            )
        }
    }

    // MARK: - Solvers

    /// Basic Euler solver with no CFG. Used for meanflow distilled models.
    /// At each step we call the decoder with `t` (current) and `r` (next).
    private func basicEuler(
        x xIn: MLXArray,
        tSpan: MLXArray,
        mu: MLXArray,
        mask: MLXArray,
        spks: MLXArray?,
        cond: MLXArray?
    ) -> MLXArray {
        var x = xIn
        let steps = tSpan.shape[0] - 1
        for i in 0..<steps {
            // t and r are shape (1,) — but the decoder broadcasts a (B,)
            // timestep. The Python ref slices `t = t_span[i:i+1]` (shape (1,)).
            // Our decoder treats t as (B,) via sinusoidalPosEmb; for B=1 the
            // (1,) shape works directly.
            let t = tSpan[i..<(i + 1)]
            let r = tSpan[(i + 1)..<(i + 2)]

            let dxdt = estimator(
                x, mask: mask, mu: mu, t: t, spks: spks, cond: cond, r: r
            )
            let dt = r - t
            x = x + dt * dxdt
        }
        return x
    }

    /// Euler solver with classifier-free guidance. Used for non-meanflow
    /// base Chatterbox. NOT exercised in Chatterbox-Turbo (meanflow=True).
    private func solveEulerCFG(
        x xIn: MLXArray,
        tSpan: MLXArray,
        mu: MLXArray,
        mask: MLXArray,
        spks: MLXArray?,
        cond: MLXArray?
    ) -> MLXArray {
        var x = xIn
        let B = mu.shape[0]
        let steps = tSpan.shape[0] - 1
        for i in 0..<steps {
            let t = tSpan[i..<(i + 1)]

            // Duplicate everything along batch dim: [cond, uncond].
            let xIn2 = concatenated([x, x], axis: 0)
            let maskIn = concatenated([mask, mask], axis: 0)
            let muIn = concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
            let tIn = MLX.broadcast(t, to: [2 * B])

            let spksIn: MLXArray? = spks.map {
                concatenated([$0, MLXArray.zeros(like: $0)], axis: 0)
            }
            let condIn: MLXArray? = cond.map {
                concatenated([$0, MLXArray.zeros(like: $0)], axis: 0)
            }

            let dxdtFull = estimator(
                xIn2, mask: maskIn, mu: muIn, t: tIn, spks: spksIn, cond: condIn
            )

            // Split: first B = conditional, second B = unconditional.
            let dxdtCond = dxdtFull[0..<B]
            let dxdtUncond = dxdtFull[B..<(2 * B)]
            let w = inferenceCfgRate
            let dxdt = (1.0 + w) * dxdtCond - w * dxdtUncond

            let dt = tSpan[(i + 1)..<(i + 2)] - t
            x = x + dt * dxdt
        }
        return x
    }
}
