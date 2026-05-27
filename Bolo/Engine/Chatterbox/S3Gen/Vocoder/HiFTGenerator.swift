// Bolo/Engine/Chatterbox/S3Gen/Vocoder/HiFTGenerator.swift
import Foundation
import MLX
import MLXNN

/// HiFTNet Generator — Neural Source Filter + ISTFTNet vocoder.
///
/// Mirrors `HiFTGenerator` in
/// `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.hifigan`. Takes a
/// mel-like feature `(B, 80, T)` and produces 24 kHz raw audio
/// `(B, T_audio)`.
///
/// Architecture overview (T = `T_mel`):
///
///     mel (B, 80, T)
///       ├─► F0Predictor → f0 (B, T)
///       │     │
///       │     └─► repeat upsample_scale (=480) → SourceModule (SineGen+l_linear)
///       │           → source (B, 1, T*480)
///       │           → STFT → s_stft (B, n_fft+2, T_frames)
///       └─► conv_pre → (B, 512, T)
///             │
///             For i in 0..<3 (upsample stages):
///               x = leaky_relu(x, 0.1) → ups[i] → x  (channels halve each step)
///               (last stage only) left-pad time axis by 1
///               si = source_resblocks[i](source_downs[i](s_stft))
///               x = x[:, :, :min] + si[:, :, :min]
///               x = avg over 3 main resblocks at this stage
///             x = leaky_relu(x) → conv_post → (B, n_fft+2, T_out)
///             magnitude = exp(x[:, :n_fft/2+1, :])
///             phase     = sin(x[:, n_fft/2+1:, :])
///             ISTFT(magnitude, phase) → audio (B, T_audio)
///
/// All Conv1d weights in the on-disk safetensors are stored MLX-style
/// `(O, K, I)` — no transposition needed.
final class HiFTGenerator: Module {

    // MARK: - Configuration

    let samplingRate: Int
    let nFFT: Int
    let hopLen: Int
    let numUpsamples: Int
    let numKernels: Int
    let f0UpsampleScale: Int
    let audioLimit: Float = 0.99

    // MARK: - Children

    @ModuleInfo(key: "f0_predictor") var f0Predictor: F0Predictor
    @ModuleInfo(key: "m_source") var mSource: SourceModule

    @ModuleInfo(key: "conv_pre") var convPre: Conv1dPT
    @ModuleInfo(key: "ups") var ups: [ConvTranspose1dPT]
    @ModuleInfo(key: "source_downs") var sourceDowns: [Conv1dPT]
    @ModuleInfo(key: "source_resblocks") var sourceResblocks: [ResBlockVocoder]
    @ModuleInfo(key: "resblocks") var resblocks: [ResBlockVocoder]
    @ModuleInfo(key: "conv_post") var convPost: Conv1dPT

    init(
        inChannels: Int = 80,
        baseChannels: Int = 512,
        nbHarmonics: Int = 8,
        samplingRate: Int = 24000,
        nsfAlpha: Float = 0.1,
        nsfSigma: Float = 0.003,
        nsfVoicedThreshold: Float = 10,
        upsampleRates: [Int] = [8, 5, 3],
        upsampleKernelSizes: [Int] = [16, 11, 7],
        resblockKernelSizes: [Int] = [3, 7, 11],
        resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        sourceResblockKernelSizes: [Int] = [7, 7, 11],
        sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        nFFT: Int = 16,
        hopLen: Int = 4
    ) {
        self.samplingRate = samplingRate
        self.nFFT = nFFT
        self.hopLen = hopLen
        self.numUpsamples = upsampleRates.count
        self.numKernels = resblockKernelSizes.count
        let upsampleProduct = upsampleRates.reduce(1, *)
        self.f0UpsampleScale = upsampleProduct * hopLen

        // F0 predictor + source module
        self._f0Predictor.wrappedValue = F0Predictor()
        self._mSource.wrappedValue = SourceModule(
            samplingRate: samplingRate,
            harmonicNum: nbHarmonics,
            sineAmp: nsfAlpha,
            addNoiseStd: nsfSigma,
            voicedThreshold: nsfVoicedThreshold
        )

        // conv_pre (in → base_channels, kernel=7, padding=3)
        self._convPre.wrappedValue = Conv1dPT(
            inputChannels: inChannels,
            outputChannels: baseChannels,
            kernelSize: 7,
            padding: 3
        )

        // Upsampling layers
        var upsList: [ConvTranspose1dPT] = []
        var ch = baseChannels
        for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
            let inC = ch / (1 << i)
            let outC = ch / (1 << (i + 1))
            upsList.append(
                ConvTranspose1dPT(
                    inputChannels: inC,
                    outputChannels: outC,
                    kernelSize: k,
                    stride: u,
                    padding: (k - u) / 2
                )
            )
            _ = ch  // silence unused warning
        }
        self._ups.wrappedValue = upsList

        // Source downsampling + resblocks
        // downsample_rates = [1] + upsample_rates[::-1][:-1]
        // i.e. for [8,5,3]: reversed_truncated = [3,5], so downsample_rates = [1,3,5]
        // downsample_cum = cumprod = [1, 3, 15]
        // The iteration is over downsample_cum[::-1] = [15, 3, 1]
        var downsampleRates: [Int] = [1]
        let upsampleReversed = upsampleRates.reversed()
        for (idx, r) in upsampleReversed.enumerated() {
            if idx < upsampleReversed.count - 1 {  // skip last
                downsampleRates.append(r)
            }
        }
        var downsampleCum: [Int] = []
        var prod = 1
        for r in downsampleRates {
            prod *= r
            downsampleCum.append(prod)
        }
        let downsampleCumReversed = Array(downsampleCum.reversed())

        var sourceDownsList: [Conv1dPT] = []
        var sourceResblocksList: [ResBlockVocoder] = []
        for (i, u) in downsampleCumReversed.enumerated() {
            let outC = baseChannels / (1 << (i + 1))
            let k = sourceResblockKernelSizes[i]
            let d = sourceResblockDilationSizes[i]
            if u == 1 {
                sourceDownsList.append(
                    Conv1dPT(
                        inputChannels: nFFT + 2,
                        outputChannels: outC,
                        kernelSize: 1
                    )
                )
            } else {
                sourceDownsList.append(
                    Conv1dPT(
                        inputChannels: nFFT + 2,
                        outputChannels: outC,
                        kernelSize: u * 2,
                        stride: u,
                        padding: u / 2
                    )
                )
            }
            sourceResblocksList.append(
                ResBlockVocoder(channels: outC, kernelSize: k, dilations: d)
            )
        }
        self._sourceDowns.wrappedValue = sourceDownsList
        self._sourceResblocks.wrappedValue = sourceResblocksList

        // Main resblocks (3 stages × 3 kernels = 9 total in production)
        var resblocksList: [ResBlockVocoder] = []
        for i in 0..<numUpsamples {
            let resCh = baseChannels / (1 << (i + 1))
            for (k, d) in zip(resblockKernelSizes, resblockDilationSizes) {
                resblocksList.append(
                    ResBlockVocoder(channels: resCh, kernelSize: k, dilations: d)
                )
            }
        }
        self._resblocks.wrappedValue = resblocksList

        // conv_post
        let finalCh = baseChannels / (1 << numUpsamples)
        self._convPost.wrappedValue = Conv1dPT(
            inputChannels: finalCh,
            outputChannels: nFFT + 2,
            kernelSize: 7,
            padding: 3
        )

        super.init()
    }

    // MARK: - Forward

    /// Inference: take a mel `(B, T, 80)` and produce audio `(B, T_audio)`.
    ///
    /// - Parameters:
    ///   - speechFeat: `(B, T, 80)` mel features (transposed layout, matching
    ///     the Python `inference()` API).
    ///   - sineGenRandomPhases: optional `(B, nbHarmonics, 1)` pre-sampled
    ///     uniform phases. Pass through for deterministic parity.
    ///   - sineGenNoise: optional `(B, nbHarmonics+1, T_audio)` standard-
    ///     normal samples for SineGen.
    /// - Returns: `(audio, source)` with `audio` shape `(B, T_audio)`.
    func inference(
        speechFeat: MLXArray,
        sineGenRandomPhases: MLXArray? = nil,
        sineGenNoise: MLXArray? = nil
    ) -> (audio: MLXArray, source: MLXArray) {
        // Transpose (B, T, 80) -> (B, 80, T) for internal compute.
        let mel = speechFeat.transposed(0, 2, 1)
        return forward(
            mel: mel,
            sineGenRandomPhases: sineGenRandomPhases,
            sineGenNoise: sineGenNoise
        )
    }

    /// Forward with mel already in `(B, 80, T)` layout.
    func forward(
        mel: MLXArray,
        sineGenRandomPhases: MLXArray? = nil,
        sineGenNoise: MLXArray? = nil
    ) -> (audio: MLXArray, source: MLXArray) {
        // 1. Predict + upsample F0.
        let f0 = f0Predictor(mel)                               // (B, T_mel)
        let f0Up = upsampleF0(f0)                               // (B, T_mel*scale, 1)

        // 2. Source module.
        let (sineMerge, _, _) = mSource(
            f0Up,
            sineGenRandomPhases: sineGenRandomPhases,
            sineGenNoise: sineGenNoise
        )
        // (B, T, 1) -> (B, 1, T) — matches Python's transpose post m_source.
        let source = sineMerge.transposed(0, 2, 1)              // (B, 1, T_audio)

        // 3. Decode.
        let audio = decode(mel: mel, source: source)
        return (audio, source)
    }

    /// Repeat upsample F0 by `f0UpsampleScale` along the time dimension.
    /// Output shape `(B, T_mel * scale, 1)`.
    func upsampleF0(_ f0: MLXArray) -> MLXArray {
        // f0: (B, T) -> (B, T, 1)
        let f0_3 = f0.expandedDimensions(axis: -1)
        return MLX.repeated(f0_3, count: f0UpsampleScale, axis: 1)
    }

    // MARK: - Decode core

    /// Run the main convolutional decoder: takes mel `(B, 80, T_mel)` and
    /// source signal `(B, 1, T_audio)`, returns audio `(B, T_audio)`.
    func decode(mel: MLXArray, source: MLXArray) -> MLXArray {
        // Compute source STFT -> (B, n_fft+2, T_frames)
        let (sReal, sImag) = stft(source[0..., 0, 0...])
        let sStft = concatenated([sReal, sImag], axis: 1)

        // conv_pre
        var x = convPre(mel)                                    // (B, base, T_mel)

        // Upsampling with source fusion
        for i in 0..<numUpsamples {
            x = leakyRelu(x, negativeSlope: 0.1)
            x = ups[i](x)

            // Reflection pad on last stage: one zero on the left of time.
            if i == numUpsamples - 1 {
                // pad axis 2 by (1, 0). pad widths = [(0,0), (0,0), (1,0)]
                x = padded(
                    x,
                    widths: [IntOrPair(0), IntOrPair(0), IntOrPair((1, 0))]
                )
            }

            // Source fusion
            var si = sourceDowns[i](sStft)
            si = sourceResblocks[i](si)

            let xT = x.shape[2]
            let siT = si.shape[2]
            let minT = min(xT, siT)
            x = x[0..., 0..., 0..<minT] + si[0..., 0..., 0..<minT]

            // Main resblocks at this stage — average
            var xs: MLXArray? = nil
            for j in 0..<numKernels {
                let idx = i * numKernels + j
                let y = resblocks[idx](x)
                xs = (xs == nil) ? y : (xs! + y)
            }
            x = xs! / Float(numKernels)
        }

        x = leakyRelu(x)
        x = convPost(x)

        // Split into magnitude and phase.
        let half = nFFT / 2 + 1
        let magLog = x[0..., 0..<half, 0...]
        let phaseRaw = x[0..., half..., 0...]
        let magnitude = MLX.exp(magLog)
        let phase = MLX.sin(phaseRaw)

        // ISTFT
        var audio = istft(magnitude: magnitude, phase: phase)
        audio = MLX.clip(audio, min: -audioLimit, max: audioLimit)
        return audio
    }

    // MARK: - LeakyReLU helper (matches PyTorch default)

    /// PyTorch default leaky_relu uses negative_slope = 0.01. The vocoder
    /// uses 0.1 explicitly inside the upsample loop and 0.01 (default) before
    /// conv_post.
    private func leakyRelu(_ x: MLXArray, negativeSlope: Float = 0.01) -> MLXArray {
        // x if x > 0 else negative_slope * x
        return MLX.which(x .> 0, x, negativeSlope * x)
    }

    // MARK: - STFT / ISTFT (numeric, n_fft=16 is tiny)

    /// Short-time Fourier transform. Returns real, imag parts each of shape
    /// `(B, n_fft/2+1, T_frames)`. Matches `np.fft.rfft` over windowed frames
    /// using a periodic Hann window.
    func stft(_ x: MLXArray) -> (real: MLXArray, imag: MLXArray) {
        let nFft = self.nFFT
        let hop = self.hopLen
        let B = x.shape[0]
        let T = x.shape[1]

        // Build the Hann window (periodic).
        let window = hannWindow(length: nFft, periodic: true)

        // Frames.
        let nFrames = max(1, (T - nFft) / hop + 1)
        var xData = x.asArray(Float.self)        // length B * T
        if (T - nFft) / hop + 1 < 1 {
            // Pad input to at least n_fft length.
            var padded = [Float](repeating: 0, count: B * nFft)
            for b in 0..<B {
                for t in 0..<T {
                    padded[b * nFft + t] = xData[b * T + t]
                }
            }
            xData = padded
        }
        let actualT = (xData.count / B)

        let halfFreq = nFft / 2 + 1
        var realOut = [Float](repeating: 0, count: B * halfFreq * nFrames)
        var imagOut = [Float](repeating: 0, count: B * halfFreq * nFrames)

        var frame = [Float](repeating: 0, count: nFft)
        for b in 0..<B {
            for f in 0..<nFrames {
                let start = f * hop
                for n in 0..<nFft {
                    frame[n] = xData[b * actualT + start + n] * window[n]
                }
                // Naive DFT (rfft) for nFft=16: ~16*9 ops per frame, fine.
                for k in 0..<halfFreq {
                    var re: Float = 0
                    var im: Float = 0
                    let coef = -2 * Float.pi * Float(k) / Float(nFft)
                    for n in 0..<nFft {
                        let angle = coef * Float(n)
                        re += frame[n] * cos(angle)
                        im += frame[n] * sin(angle)
                    }
                    // Layout (B, freq, time): index b*halfFreq*nFrames + k*nFrames + f
                    realOut[b * halfFreq * nFrames + k * nFrames + f] = re
                    imagOut[b * halfFreq * nFrames + k * nFrames + f] = im
                }
            }
        }

        let realArr = MLXArray(realOut, [B, halfFreq, nFrames])
        let imagArr = MLXArray(imagOut, [B, halfFreq, nFrames])
        return (realArr, imagArr)
    }

    /// Inverse STFT matching `torch.istft(center=True)`. Takes magnitude
    /// and phase, both `(B, n_fft/2+1, T_frames)`. Returns `(B, T_audio)`
    /// with `T_audio = (T_frames - 1) * hop_len`.
    func istft(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        let nFft = self.nFFT
        let hop = self.hopLen

        // Clip magnitude for stability (matches Python).
        let mag = MLX.clip(magnitude, max: 1e2)

        let B = mag.shape[0]
        let halfFreq = mag.shape[1]
        let T = mag.shape[2]

        // Reconstruct frames via inverse DFT (n_fft=16 -> 16 samples per frame).
        // mag*cos(phase), mag*sin(phase) -> per-frequency complex coefficients.
        let realCoef = (mag * MLX.cos(phase)).asArray(Float.self)
        let imagCoef = (mag * MLX.sin(phase)).asArray(Float.self)
        // Layout (B, halfFreq, T)

        let window = hannWindow(length: nFft, periodic: true)

        // Output buffer with overlap-add.
        let outputLength = (T - 1) * hop + nFft
        var audioBuf = [Float](repeating: 0, count: B * outputLength)
        var windowSum = [Float](repeating: 0, count: outputLength)

        // Pre-compute IDFT basis for n_fft=16.
        // x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(j*2*pi*k*n/N)
        // Using rfft convention: X[N-k] = conj(X[k]) for k in 1..N/2-1,
        // so we use only halfFreq coefficients and reconstruct via symmetry.
        // For each frame f, frame[n] = (1/N) * (X[0] + 2*Re(sum_{k=1}^{N/2-1} X[k] e^{j 2π k n/N}) + X[N/2]*(-1)^n)
        // (X[0] and X[N/2] are real.)
        // But our coefficients are general complex; treat them as full rfft form.

        var windowSumComputed = false
        for n in 0..<nFft {
            windowSum[0] += 0  // placeholder
            _ = n
        }

        var frameRecon = [Float](repeating: 0, count: nFft)
        let invN = 1.0 / Float(nFft)
        for b in 0..<B {
            for f in 0..<T {
                // Reconstruct frame from rfft coefficients.
                for n in 0..<nFft {
                    var s: Float = 0
                    for k in 0..<halfFreq {
                        let re = realCoef[b * halfFreq * T + k * T + f]
                        let im = imagCoef[b * halfFreq * T + k * T + f]
                        let angle = 2 * Float.pi * Float(k) * Float(n) / Float(nFft)
                        let c = cos(angle)
                        let si = sin(angle)
                        // For k=0 and k=N/2 just use real * cos
                        if k == 0 || k == halfFreq - 1 {
                            s += re * c - im * si
                        } else {
                            // Symmetric counterpart contributes conj(X[k]) * e^{j 2π (N-k) n / N}
                            // which equals (re*cos(-angle) - (-im)*sin(-angle)) … equivalently:
                            // 2 * Re(X[k] * e^{j angle}) = 2*(re*cos - im*sin)
                            s += 2 * (re * c - im * si)
                        }
                    }
                    frameRecon[n] = s * invN
                }

                let start = f * hop
                for n in 0..<nFft {
                    let v = frameRecon[n] * window[n]
                    audioBuf[b * outputLength + start + n] += v
                    if !windowSumComputed {
                        windowSum[start + n] += window[n] * window[n]
                    }
                }
            }
            windowSumComputed = true
        }

        // Normalize.
        for i in 0..<outputLength {
            if windowSum[i] < 1e-8 { windowSum[i] = 1e-8 }
        }
        for b in 0..<B {
            for i in 0..<outputLength {
                audioBuf[b * outputLength + i] /= windowSum[i]
            }
        }

        // Trim to match `torch.istft(center=True)`:
        //   pad = n_fft // 2; expected_len = (T - 1) * hop; output = output[pad: pad+expected_len]
        let pad = nFft / 2
        let expectedLen = (T - 1) * hop
        var trimmed = [Float](repeating: 0, count: B * expectedLen)
        for b in 0..<B {
            for i in 0..<expectedLen {
                trimmed[b * expectedLen + i] = audioBuf[b * outputLength + pad + i]
            }
        }
        return MLXArray(trimmed, [B, expectedLen])
    }

    /// Periodic Hann window of `length`. Returns Float array.
    func hannWindow(length: Int, periodic: Bool = true) -> [Float] {
        let denom = periodic ? Float(length) : Float(length - 1)
        var w = [Float](repeating: 0, count: length)
        for n in 0..<length {
            w[n] = 0.5 * (1 - cos(2 * Float.pi * Float(n) / denom))
        }
        return w
    }
}
