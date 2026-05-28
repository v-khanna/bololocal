// Bolo/UI/ReadingCapsule.swift
// Faithful port of FreeFlow's RecordingOverlay rendering (zachlatta/freeflow,
// Sources/RecordingOverlay.swift) so Bolo's reading indicator feels IDENTICAL
// to FreeFlow's recording overlay. Differences from FreeFlow are only those
// inherent to text→speech vs speech→text:
//   • Bolo always shows the stop button (TTS plays to completion or until
//     stopped; FreeFlow only shows stop in toggle-recording mode).
//   • The waveform runs its activity pulse whenever audio is playing and
//     freezes when paused (FreeFlow pulses while recording).
// No label, no icon, no timer — exactly like FreeFlow's winged overlay.

import SwiftUI

// MARK: - Waveform bars (verbatim from FreeFlow WaveformBar / CompactWaveformBar)

/// Full-size bar — used by the centered pill on non-notched displays.
struct BoloWaveformBar: View {
    let amplitude: CGFloat
    private let minHeight: CGFloat = 2
    private let maxHeight: CGFloat = 22
    var body: some View {
        Capsule()
            .fill(.white)
            .frame(width: 3, height: minHeight + (maxHeight - minHeight) * amplitude)
    }
}

/// Tighter bar — used inside the 36pt wing on notched displays.
struct BoloCompactWaveformBar: View {
    let amplitude: CGFloat
    private let minHeight: CGFloat = 2
    private let maxHeight: CGFloat = 14
    var body: some View {
        Capsule()
            .fill(.white)
            .frame(width: 2, height: minHeight + (maxHeight - minHeight) * amplitude)
    }
}

// MARK: - Waveform views (verbatim pulse math from FreeFlow)

/// 9-bar waveform for the centered pill (non-notched displays).
struct BoloWaveformView: View {
    /// 0…1 live level. Bolo drives the pulse purely from time (level stays 0),
    /// which is exactly how FreeFlow renders when the mic is silent.
    var audioLevel: Float = 0
    var showsActivityPulse: Bool

    private static let barCount = 9
    private static let multipliers: [CGFloat] = [0.35, 0.55, 0.75, 0.9, 1.0, 0.9, 0.75, 0.55, 0.35]
    private static let centerIndex = CGFloat((barCount - 1) / 2)

    var body: some View {
        Group {
            if showsActivityPulse {
                TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { context in
                    bars(pulseTime: context.date.timeIntervalSinceReferenceDate)
                }
            } else {
                bars(pulseTime: nil)
            }
        }
        .frame(height: 24)
    }

    private func bars(pulseTime: TimeInterval?) -> some View {
        HStack(spacing: 2.5) {
            ForEach(0..<Self.barCount, id: \.self) { index in
                BoloWaveformBar(amplitude: amplitude(for: index, pulseTime: pulseTime))
                    .animation(
                        .spring(response: barResponse(for: index), dampingFraction: 0.88)
                            .delay(barDelay(for: index)),
                        value: audioLevel
                    )
            }
        }
    }

    private func amplitude(for index: Int, pulseTime: TimeInterval?) -> CGFloat {
        let level = CGFloat(max(audioLevel, 0))
        let base = min(level * Self.multipliers[index], 1.0)
        guard let pulseTime else { return base }
        let traveling = CGFloat(0.5 + 0.5 * sin((pulseTime * 6.2) - Double(index) * 0.78))
        let shimmer = CGFloat(0.5 + 0.5 * sin((pulseTime * 3.1) + Double(index) * 0.5))
        let pulse = traveling * 0.22 + shimmer * 0.06
        let saturationRelief = base * (0.74 + pulse)
        let quietPulse = (1.0 - base) * (0.04 + pulse * 0.28)
        return min(saturationRelief + quietPulse, 1.0)
    }

    private func barResponse(for index: Int) -> Double {
        let distance = abs(CGFloat(index) - Self.centerIndex)
        let normalizedDistance = distance / Self.centerIndex
        return 0.18 + Double(normalizedDistance) * 0.06
    }

    private func barDelay(for index: Int) -> Double {
        let distance = abs(CGFloat(index) - Self.centerIndex)
        return Double(distance) * 0.01
    }
}

/// 5-bar waveform sized for the 36pt wing layout (notched displays).
struct BoloCompactWaveformView: View {
    var audioLevel: Float = 0
    var showsActivityPulse: Bool

    private static let barCount = 5
    private static let multipliers: [CGFloat] = [0.5, 0.75, 1.0, 0.75, 0.5]

    var body: some View {
        Group {
            if showsActivityPulse {
                TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { context in
                    bars(pulseTime: context.date.timeIntervalSinceReferenceDate)
                }
            } else {
                bars(pulseTime: nil)
            }
        }
        .frame(height: 18)
    }

    private func bars(pulseTime: TimeInterval?) -> some View {
        HStack(spacing: 1.5) {
            ForEach(0..<Self.barCount, id: \.self) { index in
                BoloCompactWaveformBar(amplitude: amplitude(for: index, pulseTime: pulseTime))
                    .animation(.spring(response: 0.18, dampingFraction: 0.88), value: audioLevel)
            }
        }
    }

    private func amplitude(for index: Int, pulseTime: TimeInterval?) -> CGFloat {
        let level = CGFloat(max(audioLevel, 0))
        let base = min(level * Self.multipliers[index], 1.0)
        guard let pulseTime else { return base }
        let traveling = CGFloat(0.5 + 0.5 * sin((pulseTime * 6.2) - Double(index) * 0.78))
        let shimmer = CGFloat(0.5 + 0.5 * sin((pulseTime * 3.1) + Double(index) * 0.5))
        let pulse = traveling * 0.22 + shimmer * 0.06
        let saturationRelief = base * (0.74 + pulse)
        let quietPulse = (1.0 - base) * (0.04 + pulse * 0.28)
        return min(saturationRelief + quietPulse, 1.0)
    }
}

// MARK: - Stop button (verbatim style from FreeFlow)

struct BoloStopButton: View {
    let onStop: () -> Void
    var body: some View {
        Button(action: onStop) {
            Image(systemName: "stop.fill")
                .font(.system(size: 7, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 14, height: 14)
                .background(Circle().fill(Color.red.opacity(0.92)))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Winged layout (notched displays) — FreeFlow WingedRecordingView

/// `[ waveform wing | solid-black notch spacer | stop wing ]`. The black notch
/// spacer sits under the camera cutout so the hardware masks it, making the
/// overlay read as an extension of the notch.
struct WingedReadingView: View {
    @ObservedObject var coordinator: CoordinatorState
    let leftWingWidth: CGFloat
    let notchWidth: CGFloat
    let rightWingWidth: CGFloat
    let height: CGFloat
    let onStop: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            // Left wing — waveform, centered.
            HStack {
                Spacer(minLength: 0)
                BoloCompactWaveformView(showsActivityPulse: coordinator.isPlaying)
                Spacer(minLength: 0)
            }
            .frame(width: leftWingWidth, height: height)

            // Notch spacer — solid black; camera cutout hides it.
            Color.black
                .frame(width: notchWidth, height: height)

            // Right wing — stop button, centered.
            HStack {
                Spacer(minLength: 0)
                BoloStopButton(onStop: onStop)
                Spacer(minLength: 0)
            }
            .frame(width: rightWingWidth, height: height)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Centered pill (non-notched displays)

/// Compact centered pill for flat displays: waveform + stop, padded.
struct PillReadingView: View {
    @ObservedObject var coordinator: CoordinatorState
    let onStop: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            BoloWaveformView(showsActivityPulse: coordinator.isPlaying)
            BoloStopButton(onStop: onStop)
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Brand glyph (speech bubble + play) — kept for the status item / future use

/// Speech bubble + play triangle, authored in an 18×18 viewBox. Not used by the
/// winged capsule (which matches FreeFlow's label-free look) but retained for
/// the menu-bar status icon and Settings/About.
struct BoloGlyph: Shape {
    var filled: Bool = true
    func path(in rect: CGRect) -> Path {
        let s = min(rect.width, rect.height) / 18.0
        let o = CGPoint(x: rect.midX - 9 * s, y: rect.midY - 9 * s)
        func p(_ x: Double, _ y: Double) -> CGPoint { CGPoint(x: o.x + CGFloat(x) * s, y: o.y + CGFloat(y) * s) }
        var path = Path()
        path.move(to: p(14.5, 3.5))
        path.addLine(to: p(3.5, 3.5))
        path.addArc(tangent1End: p(2, 3.5), tangent2End: p(2, 5), radius: 1.5 * s)
        path.addLine(to: p(2, 11))
        path.addArc(tangent1End: p(2, 12.5), tangent2End: p(3.5, 12.5), radius: 1.5 * s)
        path.addLine(to: p(6, 12.5))
        path.addLine(to: p(6, 15))
        path.addLine(to: p(9, 12.5))
        path.addLine(to: p(14.5, 12.5))
        path.addArc(tangent1End: p(16, 12.5), tangent2End: p(16, 10.5), radius: 1.5 * s)
        path.addLine(to: p(16, 5))
        path.addArc(tangent1End: p(16, 3.5), tangent2End: p(14.5, 3.5), radius: 1.5 * s)
        path.closeSubpath()
        if filled {
            path.move(to: p(7.6, 6.2))
            path.addLine(to: p(7.6, 10.8))
            path.addLine(to: p(11.4, 8.5))
            path.closeSubpath()
        }
        return path
    }
}
