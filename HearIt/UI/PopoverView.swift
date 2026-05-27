import SwiftUI

struct PopoverView: View {
    @ObservedObject var settings: Settings
    @ObservedObject var coordinator: CoordinatorState
    let onOpenSettings: () -> Void

    var body: some View {
        ZStack {
            VisualEffectBackground()
            VStack(alignment: .leading, spacing: 12) {
                header
                textPreview
                Divider()
                controls
                Divider()
                speedSlider
                Spacer()
            }
            .padding(16)
        }
        .frame(width: 320, height: 320)
    }

    private var header: some View {
        HStack {
            Image(systemName: "waveform")
                .foregroundStyle(.secondary)
            Text("HearIt").font(.headline)
            Spacer()
            Button(action: onOpenSettings) {
                Image(systemName: "gearshape").foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .keyboardShortcut(",", modifiers: .command)
            .help("Settings")
        }
    }

    private var textPreview: some View {
        ScrollView {
            Text(coordinator.lastCapturedText.isEmpty
                 ? "Select text in any app and press ⌘⇧R."
                 : coordinator.lastCapturedText)
                .font(.callout)
                .foregroundStyle(coordinator.lastCapturedText.isEmpty ? .secondary : .primary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
        }
        .frame(height: 80)
    }

    private var controls: some View {
        HStack(spacing: 18) {
            Button(action: coordinator.togglePlayPause) {
                Image(systemName: coordinator.isPlaying ? "pause.fill" : "play.fill")
                    .font(.title2)
            }
            .buttonStyle(.plain)
            .disabled(coordinator.lastCapturedText.isEmpty)

            Button(action: coordinator.stop) {
                Image(systemName: "stop.fill").font(.title2)
            }
            .buttonStyle(.plain)
            .disabled(!coordinator.isPlaying)

            Spacer()
        }
    }

    private var speedSlider: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Speed").foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1fx", settings.speed.value))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Slider(value: Binding(
                get: { settings.speed.value },
                set: { settings.speed = Speed($0) }
            ), in: 0.5...2.0, step: 0.1)
        }
    }
}

/// Observable mirror of Coordinator playback state — kept thin to avoid SwiftUI/AppKit threading mess.
@MainActor
final class CoordinatorState: ObservableObject {
    @Published var lastCapturedText: String = ""
    @Published var isPlaying: Bool = false
    var togglePlayPause: () -> Void = {}
    var stop: () -> Void = {}
}
