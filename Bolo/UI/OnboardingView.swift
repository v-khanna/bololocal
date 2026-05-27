import SwiftUI

struct OnboardingView: View {
    @State private var step: Step = .welcome
    @ObservedObject var downloadProgress: ModelDownloadProgress
    let onStartDownload: () -> Void
    let onComplete: () -> Void

    enum Step { case welcome, accessibility, modelDownload, ready }

    var body: some View {
        ZStack {
            VisualEffectBackground()
            content
                .padding(32)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(width: 520, height: 380)
    }

    @ViewBuilder
    private var content: some View {
        switch step {
        case .welcome:        welcome
        case .accessibility:  accessibility
        case .modelDownload:  modelDownload
        case .ready:          ready
        }
    }

    private var welcome: some View {
        VStack(spacing: 16) {
            Image(systemName: "waveform")
                .font(.system(size: 64))
                .foregroundStyle(.tint)
            Text("Welcome to Bolo").font(.title)
            Text("Select text in any app, press ⌘⇧R, and hear it read aloud in a natural AI voice. Fully on-device.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 24)
            Spacer()
            Button("Continue") { step = .accessibility }
                .keyboardShortcut(.defaultAction)
                .controlSize(.large)
        }
    }

    private var accessibility: some View {
        VStack(spacing: 16) {
            Image(systemName: "lock.shield")
                .font(.system(size: 64))
                .foregroundStyle(.tint)
            Text("Allow Accessibility").font(.title2)
            Text("Bolo needs Accessibility access to read text you select in other apps. Nothing leaves your Mac.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 24)
            Button("Open System Settings") {
                PermissionsManager.requestAccessibility()
                PermissionsManager.openAccessibilitySettings()
            }
            .controlSize(.large)
            Spacer()
            Button(PermissionsManager.isAccessibilityGranted ? "Continue" : "I've enabled it") {
                if PermissionsManager.isAccessibilityGranted {
                    step = .modelDownload
                    onStartDownload()
                }
            }
            .disabled(!PermissionsManager.isAccessibilityGranted)
            .keyboardShortcut(.defaultAction)
        }
    }

    private var modelDownload: some View {
        VStack(spacing: 16) {
            Image(systemName: downloadProgress.isComplete ? "checkmark.circle.fill" : "arrow.down.circle")
                .font(.system(size: 64))
                .foregroundStyle(downloadProgress.isComplete ? Color.green : Color.accentColor)
            Text(downloadProgress.isComplete ? "Voice ready" : "Downloading voice model")
                .font(.title2)
            Text("Bolo is downloading the Qwen3-TTS model — about 500 MB. This happens once.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 24)
            ProgressView(value: downloadProgress.progress)
                .frame(width: 360)
                .progressViewStyle(.linear)
            Text(downloadProgress.label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(width: 360)
            if let err = downloadProgress.error {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                Button("Retry") {
                    downloadProgress.reset()
                    onStartDownload()
                }
            }
            Spacer()
            Button("Continue") { step = .ready }
                .disabled(!downloadProgress.isComplete)
                .keyboardShortcut(.defaultAction)
                .controlSize(.large)
        }
    }

    private var ready: some View {
        VStack(spacing: 16) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 64))
                .foregroundStyle(.green)
            Text("You're set").font(.title2)
            Text("Select text in any app and press ⌘⇧R to hear it read aloud. Bolo lives in your menu bar.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 24)
            Spacer()
            Button("Get Started") { onComplete() }
                .keyboardShortcut(.defaultAction)
                .controlSize(.large)
        }
    }
}
