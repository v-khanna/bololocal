import SwiftUI
import KeyboardShortcuts

// ════════════════════════════════════════════════════════════
//  Reusable card primitives (Cotypist-style grouped cards)
// ════════════════════════════════════════════════════════════

/// Gray, semibold, slightly-tracked caption above a group of cards (.grp-cap).
struct GroupCaption: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text)
            .font(.system(size: 12.5, weight: .semibold))
            .foregroundStyle(Color(nsColor: .secondaryLabelColor))
            .padding(.horizontal, 4)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A rounded white card containing rows, with hairline dividers between them (.grp).
struct CardGroup<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        VStack(spacing: 0) {
            _VariadicView.Tree(DividedRows()) { content }
        }
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color(nsColor: .separatorColor).opacity(0.6), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

/// Lays out a card's child rows with a hairline divider between each (.grp .row + .row).
private struct DividedRows: _VariadicView.MultiViewRoot {
    func body(children: _VariadicView.Children) -> some View {
        let last = children.last?.id
        ForEach(children) { child in
            child
            if child.id != last {
                Divider()
                    .overlay(Color(nsColor: .separatorColor).opacity(0.5))
                    .padding(.leading, 14)
            }
        }
    }
}

/// A single row inside a card: title (+ optional subtitle) on the left, trailing control.
struct SettingsRow<Trailing: View>: View {
    let title: String
    var subtitle: String? = nil
    var titleColor: Color = Color(nsColor: .labelColor)
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.system(size: 13))
                    .foregroundStyle(titleColor)
                if let subtitle {
                    Text(subtitle)
                        .font(.system(size: 12))
                        .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            trailing
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(minHeight: 40)
    }
}

/// A scroll container for the right-hand content area with consistent padding/spacing.
struct PaneContent<Content: View>: View {
    var horizontalPadding: CGFloat = 22
    var verticalPadding: CGFloat = 18
    @ViewBuilder var content: Content
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                content
            }
            .padding(.horizontal, horizontalPadding)
            .padding(.vertical, verticalPadding)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

// MARK: - Small shared bits

/// The Bolo brand mark: speech bubble + embedded play triangle (the "Adopted" glyph).
struct BoloMark: View {
    var size: CGFloat = 64
    var color: Color = Color(red: 0.00, green: 0.48, blue: 1.00)

    var body: some View {
        RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
            .fill(color)
            .frame(width: size, height: size)
            .overlay(
                Image(systemName: "play.bubble.fill")
                    .font(.system(size: size * 0.5))
                    .foregroundStyle(.white)
            )
            .shadow(color: .black.opacity(0.12), radius: 2, y: 1)
    }
}

/// A small waveform glyph used in voice rows (5 vertical bars).
struct MiniWaveform: View {
    var color: Color = .white
    var barWidth: CGFloat = 2
    var height: CGFloat = 16
    private let heights: [CGFloat] = [0.4, 0.7, 1.0, 0.7, 0.4]
    var body: some View {
        HStack(spacing: 1.5) {
            ForEach(heights.indices, id: \.self) { i in
                Capsule().fill(color)
                    .frame(width: barWidth, height: height * heights[i])
            }
        }
        .frame(height: height)
    }
}

/// Tasteful "Coming soon" placeholder card matching the visual language.
struct ComingSoonPane: View {
    let symbol: String
    let tileColor: Color
    let title: String
    let blurb: String

    var body: some View {
        PaneContent {
            CardGroup {
                VStack(spacing: 12) {
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(tileColor.opacity(0.14))
                        .frame(width: 56, height: 56)
                        .overlay(
                            Image(systemName: symbol)
                                .font(.system(size: 24, weight: .medium))
                                .foregroundStyle(tileColor)
                        )
                    Text(title)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(Color(nsColor: .labelColor))
                    Text(blurb)
                        .font(.system(size: 12.5))
                        .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 360)
                    Text("Coming soon")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 3)
                        .background(
                            Capsule().fill(Color(nsColor: .labelColor).opacity(0.06))
                        )
                        .padding(.top, 2)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 28)
                .padding(.horizontal, 20)
            }
        }
    }
}

// ════════════════════════════════════════════════════════════
//  General pane
// ════════════════════════════════════════════════════════════

struct GeneralPane: View {
    @ObservedObject var settings: Settings

    /// v1 ships English; the rest of Qwen3's languages are listed but English is the
    /// only verified entry. Order/values mirror Qwen3TTSEngine.languageString(for:).
    private let languages: [(value: String, label: String)] = [
        ("english", "English (United States)"),
        ("chinese", "Chinese"),
        ("japanese", "Japanese"),
        ("korean", "Korean"),
        ("russian", "Russian"),
        ("german", "German"),
        ("french", "French"),
        ("italian", "Italian"),
        ("spanish", "Spanish"),
        ("portuguese", "Portuguese"),
    ]

    private let speedOptions: [Double] = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    var body: some View {
        PaneContent {
            GroupCaption("Voice engine")
            CardGroup {
                SettingsRow(
                    title: "Use on-device voice",
                    subtitle: "Private & offline, but downloads a multi-GB model on first use. Off = fast cloud voice."
                ) {
                    Toggle("", isOn: $settings.useLocalEngine)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
                if !settings.useLocalEngine {
                    SettingsRow(
                        title: "Groq API key",
                        subtitle: "Required for the cloud voice. Free key at console.groq.com."
                    ) {
                        SecureField("gsk_…", text: $settings.groqAPIKey)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 220)
                    }
                    SettingsRow(title: "Cloud voice") {
                        Picker("", selection: $settings.cloudVoice) {
                            ForEach(["troy", "hannah", "austin"], id: \.self) {
                                Text($0.capitalized).tag($0)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .fixedSize()
                    }
                }
            }
            Text("Switching the engine takes effect after you restart Bolo.")
                .font(.system(size: 11))
                .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                .padding(.leading, 4)

            CardGroup {
                SettingsRow(title: "Launch automatically at login") {
                    Toggle("", isOn: $settings.launchAtLogin)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .onChange(of: settings.launchAtLogin) { _, new in
                            LaunchAtLogin.set(enabled: new)
                        }
                }
                SettingsRow(
                    title: "Show floating capsule while reading",
                    subtitle: "Display the small playback capsule at the top of the screen during playback."
                ) {
                    Toggle("", isOn: $settings.showCapsuleWhileReading)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
            }

            GroupCaption("Reading")
            CardGroup {
                SettingsRow(
                    title: "Read selected text",
                    subtitle: "Press this combo with text selected in any app."
                ) {
                    KeyboardShortcuts.Recorder(for: .readSelection)
                        .frame(width: 170)
                }
                SettingsRow(
                    title: "Default speed",
                    subtitle: "You can change this per session in the menu bar dropdown."
                ) {
                    Picker("", selection: speedBinding) {
                        ForEach(speedOptions, id: \.self) { v in
                            Text(speedLabel(v)).tag(v)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .fixedSize()
                }
                SettingsRow(title: "Language") {
                    Picker("", selection: $settings.selectedLanguage) {
                        ForEach(languages, id: \.value) { lang in
                            Text(lang.label).tag(lang.value)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .fixedSize()
                }
            }

            GroupCaption("Privacy")
            CardGroup {
                SettingsRow(
                    title: "Accessibility access",
                    subtitle: "Required to read selected text from other apps."
                ) {
                    if PermissionsManager.isAccessibilityGranted {
                        HStack(spacing: 8) {
                            Label("Granted", systemImage: "checkmark")
                                .labelStyle(.titleAndIcon)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(Color(red: 0.20, green: 0.78, blue: 0.35))
                            Button("Manage…") { PermissionsManager.openAccessibilitySettings() }
                        }
                    } else {
                        Button("Grant Access…") { PermissionsManager.openAccessibilitySettings() }
                            .buttonStyle(.borderedProminent)
                    }
                }
            }
        }
    }

    private var speedBinding: Binding<Double> {
        Binding(
            get: { settings.speed.value },
            set: { settings.speed = Speed($0) }
        )
    }

    private func speedLabel(_ v: Double) -> String {
        v == v.rounded() ? String(format: "%.1f×", v) : String(format: "%.2f×", v)
    }
}

// ════════════════════════════════════════════════════════════
//  Voices pane
// ════════════════════════════════════════════════════════════

struct VoicesPane: View {
    @ObservedObject var settings: Settings

    /// Bolo's engine selects a *language*, not a named persona. We present the
    /// installed/available languages in the catalog style, blue-check on the active one.
    private struct VoiceEntry: Identifiable {
        let id: String
        let name: String
        let desc: String
        let lang: String
        let color: Color
        let installed: Bool
    }

    private let voices: [VoiceEntry] = [
        VoiceEntry(id: "english", name: "Aria", desc: "Neutral · clear", lang: "en-US",
                   color: Color(red: 0.00, green: 0.48, blue: 1.00), installed: true),
        VoiceEntry(id: "chinese", name: "Mei", desc: "Warm · expressive", lang: "zh-CN",
                   color: Color(red: 0.35, green: 0.78, blue: 0.98), installed: false),
        VoiceEntry(id: "japanese", name: "Haru", desc: "Bright · lively", lang: "ja-JP",
                   color: Color(red: 1.00, green: 0.58, blue: 0.00), installed: false),
        VoiceEntry(id: "spanish", name: "Lucia", desc: "Storyteller", lang: "es-ES",
                   color: Color(red: 0.68, green: 0.32, blue: 0.87), installed: false),
    ]

    private var selectedID: String { settings.selectedLanguage }

    var body: some View {
        PaneContent {
            GroupCaption("Selected voice")
            CardGroup {
                if let current = voices.first(where: { $0.id == selectedID }) ?? voices.first {
                    HStack(alignment: .center, spacing: 12) {
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .fill(current.color)
                            .frame(width: 38, height: 38)
                            .overlay(MiniWaveform(barWidth: 2.4, height: 18))
                        VStack(alignment: .leading, spacing: 1) {
                            Text(current.name)
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(Color(nsColor: .labelColor))
                            Text("\(current.desc) · \(current.lang)")
                                .font(.system(size: 12))
                                .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                        }
                        Spacer(minLength: 8)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 11)
                }
            }

            GroupCaption("Available voices")
            CardGroup {
                ForEach(voices) { voice in
                    voiceRow(voice)
                }
            }

            HStack(spacing: 8) {
                Text("All voices run locally on this Mac.")
                    .font(.system(size: 12))
                    .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                Spacer()
            }
        }
    }

    @ViewBuilder
    private func voiceRow(_ voice: VoiceEntry) -> some View {
        let isSelected = voice.id == selectedID
        HStack(alignment: .center, spacing: 12) {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(voice.color)
                .frame(width: 26, height: 26)
                .overlay(MiniWaveform(barWidth: 1.6, height: 12))
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(voice.name)
                        .font(.system(size: 13))
                        .foregroundStyle(Color(nsColor: .labelColor))
                    if isSelected {
                        Text("· Selected")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(Color(red: 0.20, green: 0.78, blue: 0.35))
                    }
                }
                Text("\(voice.desc) · \(voice.lang)")
                    .font(.system(size: 12))
                    .foregroundStyle(Color(nsColor: .secondaryLabelColor))
            }
            Spacer(minLength: 8)
            if isSelected {
                Image(systemName: "checkmark")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Color(red: 0.00, green: 0.48, blue: 1.00))
            } else {
                Button("Select") { settings.selectedLanguage = voice.id }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
    }
}

// ════════════════════════════════════════════════════════════
//  About pane
// ════════════════════════════════════════════════════════════

struct AboutPane: View {
    private var versionLine: String {
        let short = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "Version \(short) (build \(build)) · Apple Silicon"
    }

    var body: some View {
        PaneContent(verticalPadding: 26) {
            HStack(alignment: .center, spacing: 16) {
                BoloMark(size: 64)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Bolo")
                        .font(.system(size: 24, weight: .semibold))
                    HStack(spacing: 6) {
                        Text("बोलो")
                            .font(.system(size: 20))
                            .foregroundStyle(Color(nsColor: .labelColor).opacity(0.85))
                        Text("— \u{201C}speak\u{201D} in Hindi")
                            .font(.system(size: 12))
                            .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                    }
                    Text(versionLine)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                }
                Spacer(minLength: 8)
            }

            GroupCaption("Privacy")
            CardGroup {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Bolo runs entirely on your Mac. Text you select never leaves the device.")
                    Text("No accounts, no analytics, no telemetry. The voice model was downloaded once over HTTPS and is verified locally.")
                    Text("The app does not contact the network during normal operation.")
                }
                .font(.system(size: 12.5))
                .foregroundStyle(Color(nsColor: .labelColor).opacity(0.85))
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
            }

            GroupCaption("Engine")
            CardGroup {
                SettingsRow(title: "Voice engine") {
                    Text("Chatterbox-Turbo · MLX-Swift")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                }
            }
        }
    }
}
