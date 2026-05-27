// Bolo/Engine/Chatterbox/S3Gen/EspnetRelPositionalEncoding.swift
import Foundation
import MLX
import MLXNN

/// ESPnet-style relative positional encoding used by the S3Gen UpsampleConformer.
///
/// Mirrors `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.encoder.EspnetRelPositionalEncoding`.
///
/// On forward, scales the input by sqrt(d_model) and returns a positional embedding
/// of shape `(1, 2T-1, D)` containing sinusoidal encodings for positive AND negative
/// relative positions (positive in reverse order, negative in forward order):
///
///   pe = concat([reverse(pe_positive_0..T-1), pe_negative_1..T-1], axis=0)
///
/// This is a buffer (not a learned parameter), so it carries no entries in the
/// safetensors. It is constructed lazily on first use up to the requested sequence
/// length and cached.
///
/// Implemented as a plain class (not `Module`) since there are no learned
/// parameters to track.
final class EspnetRelPositionalEncoding {

    let dModel: Int
    let xScale: Float
    private var pe: MLXArray
    private var peLen: Int  // currently-cached size (in "T" units; pe has 2*peLen-1 rows)

    /// - Parameters:
    ///   - dModel: hidden dimension (512 for S3Gen).
    ///   - maxLen: initial table size; will be extended automatically if a longer
    ///             sequence is encountered.
    init(dModel: Int, maxLen: Int = 5000) {
        precondition(dModel % 2 == 0, "EspnetRelPositionalEncoding requires even dModel")
        self.dModel = dModel
        self.xScale = sqrt(Float(dModel))
        self.pe = MLXArray(0.0)
        self.peLen = 0
        extend(to: maxLen)
    }

    /// Build (or rebuild) the positional encoding table to cover `size` positions.
    ///
    /// The resulting table has shape `(1, 2*size - 1, dModel)`. Indexing into it
    /// at offset `center = peLen - 1` yields position 0; smaller indices are
    /// positive relative positions, larger indices are negative ones.
    private func extend(to size: Int) {
        guard size > peLen else { return }

        let halfDim = dModel / 2
        // div_term = exp(arange(0, dModel, 2) * -log(10000) / dModel)  → (halfDim,)
        let evens = MLXArray((0..<halfDim).map { Float($0 * 2) })
        let divTerm = MLX.exp(evens * Float(-log(10000.0) / Double(dModel)))

        // position: (size, 1)
        let position = MLXArray((0..<size).map { Float($0) }).reshaped([size, 1])

        // arg: (size, halfDim) = position * divTerm[None, :]
        let arg = position * divTerm.reshaped([1, halfDim])

        // Interleave sin/cos along the last axis. stacked([..., axis=-1]) gives
        // (size, halfDim, 2), then reshape to (size, dModel).
        let pePositive = MLX.stacked([MLX.sin(arg), MLX.cos(arg)], axis: -1)
            .reshaped([size, dModel])

        // Reverse along axis 0 to get position +(size-1), +(size-2), ..., 0
        let pePositiveReversed = pePositive[.stride(by: -1), .ellipsis]

        // Negative positions (we'll drop row 0 of this to avoid duplicating position 0).
        let peNegativeAll = MLX.stacked([MLX.sin(-arg), MLX.cos(-arg)], axis: -1)
            .reshaped([size, dModel])
        let peNegativeTail = peNegativeAll[1..<size, 0...]

        let table = concatenated([pePositiveReversed, peNegativeTail], axis: 0)
        pe = table.expandedDimensions(axis: 0)              // (1, 2*size-1, dModel)
        peLen = size
        eval(pe)
    }

    /// Apply scaling and return the positional embedding slice for the current sequence.
    ///
    /// - Parameter x: input `(B, T, D)`.
    /// - Returns:
    ///   - scaled input `x * sqrt(D)` (same shape as input).
    ///   - positional embedding `(1, 2T-1, D)` centered around the current position.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let T = x.shape[1]
        extend(to: T)
        let scaled = x * xScale
        let center = pe.shape[1] / 2          // == peLen - 1
        let posEmb = pe[0..., (center - T + 1)..<(center + T), 0...]  // (1, 2T-1, D)
        return (scaled, posEmb)
    }
}
