import SwiftUI

/// Cotypist-style settings: colorful tile sidebar on the left, grouped white cards on
/// the right. Entry point unchanged — `SettingsView(settings: Settings.shared)`.
struct SettingsView: View {
    @ObservedObject var settings: Settings
    @State private var selection: SettingsSection = .general

    var body: some View {
        HStack(spacing: 0) {
            SettingsSidebar(selection: $selection)
            Divider()
                .overlay(Color(nsColor: .separatorColor).opacity(0.6))
            pane
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(width: 720, height: 480)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    @ViewBuilder
    private var pane: some View {
        switch selection {
        case .general:
            GeneralPane(settings: settings)
        case .voices:
            VoicesPane(settings: settings)
        case .about:
            AboutPane()
        case .setup:
            ComingSoonPane(
                symbol: "lock.fill",
                tileColor: SettingsSection.setup.tileColor,
                title: "Setup & Recovery",
                blurb: "Re-check permissions, verify the voice model, and reset Bolo if something goes wrong. This panel is on the way."
            )
        case .shortcuts:
            ComingSoonPane(
                symbol: "command",
                tileColor: SettingsSection.shortcuts.tileColor,
                title: "Shortcuts",
                blurb: "Customize playback hotkeys — skip sentence, pause/resume, and stop. For now, set the read-selection hotkey in General."
            )
        case .appSettings:
            ComingSoonPane(
                symbol: "slider.horizontal.3",
                tileColor: SettingsSection.appSettings.tileColor,
                title: "App Settings",
                blurb: "Per-app overrides for voice and speed. Use a slower voice in PDFs, a different one in Mail — defaults apply everywhere else."
            )
        case .pronunciation:
            ComingSoonPane(
                symbol: "character.book.closed.fill",
                tileColor: SettingsSection.pronunciation.tileColor,
                title: "Pronunciation",
                blurb: "Teach Bolo how to say names, acronyms, and technical terms. Overrides will apply across all voices."
            )
        }
    }
}
