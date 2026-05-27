// Bolo/Engine/Chatterbox/T3/T3Cache.swift
import Foundation
import MLX

/// Append-only KV cache for T3's autoregressive generation.
///
/// Each call to update() appends the new (single- or multi-token) keys and values
/// to the stored history and returns the FULL cached K and V (history + new).
///
/// Shapes:
///   keys:   (B, numHeads, S_new, headDim)
///   values: (B, numHeads, S_new, headDim)
///   returns: cached (B, numHeads, S_total, headDim) for both K and V
///
/// Not thread-safe by design — lifetime is one autoregressive generation pass,
/// which is single-threaded inside the Qwen3TTSEngine actor (and will be the
/// same for ChatterboxTTSEngine when wired up in Task 12).
final class T3Cache {
    private let numHeads: Int
    private let headDim: Int

    private var storedKeys: MLXArray?
    private var storedValues: MLXArray?

    init(numHeads: Int, headDim: Int) {
        self.numHeads = numHeads
        self.headDim = headDim
    }

    /// Append new K, V slices. Returns the full cached tensors.
    /// - keys, values: shape (B, numHeads, S_new, headDim)
    /// - returns: (cachedKeys, cachedValues) both of shape (B, numHeads, S_total, headDim)
    func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        if let prevK = storedKeys, let prevV = storedValues {
            let catK = concatenated([prevK, newK], axis: 2)
            let catV = concatenated([prevV, newV], axis: 2)
            storedKeys = catK
            storedValues = catV
        } else {
            storedKeys = newK
            storedValues = newV
        }
        return (storedKeys!, storedValues!)
    }

    /// Current cached sequence length (third dimension of stored keys).
    var sequenceLength: Int {
        guard let k = storedKeys else { return 0 }
        return k.shape[2]
    }

    /// Drop the cached state. Next update() starts fresh.
    func reset() {
        storedKeys = nil
        storedValues = nil
    }
}
