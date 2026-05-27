import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: Settings

    var body: some View {
        TabView {
            generalTab.tabItem { Label("General", systemImage: "gearshape") }
            aboutTab.tabItem { Label("About", systemImage: "info.circle") }
        }
        .frame(width: 460, height: 320)
        .padding()
    }

    // MARK: - General
    private var generalTab: some View {
        Form {
            Toggle("Launch Bolo at login", isOn: $settings.launchAtLogin)
                .onChange(of: settings.launchAtLogin) { _, new in
                    LaunchAtLogin.set(enabled: new)
                }

            Picker("Language", selection: $settings.selectedLanguage) {
                Text("English").tag("english")
                // More languages added in v1.1+ as we verify voice quality per language.
            }

            HStack {
                Text("Hotkey")
                Spacer()
                Text("⌘⇧R").foregroundStyle(.secondary).monospaced()
            }

            Spacer()
            accessibilitySection
        }
    }

    private var accessibilitySection: some View {
        GroupBox {
            HStack {
                Image(systemName: PermissionsManager.isAccessibilityGranted ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(PermissionsManager.isAccessibilityGranted ? .green : .orange)
                Text(PermissionsManager.isAccessibilityGranted
                     ? "Accessibility access granted"
                     : "Accessibility access required")
                Spacer()
                if !PermissionsManager.isAccessibilityGranted {
                    Button("Open System Settings") {
                        PermissionsManager.openAccessibilitySettings()
                    }
                }
            }
            .padding(8)
        }
    }

    // MARK: - About
    private var aboutTab: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Bolo").font(.title2)
            Text("Version \(Bundle.main.shortVersionString) (\(Bundle.main.buildNumber))")
                .foregroundStyle(.secondary)
            Text("Reads selected text aloud, fully on-device.")
                .font(.caption)
            Spacer()
            Text("All speech runs locally on your Mac. No network calls after the one-time model download.")
                .multilineTextAlignment(.center)
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 24)
            Spacer()
        }
        .padding()
    }
}

private extension Bundle {
    var shortVersionString: String { infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0" }
    var buildNumber: String { infoDictionary?["CFBundleVersion"] as? String ?? "0" }
}
