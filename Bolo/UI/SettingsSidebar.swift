import SwiftUI

// MARK: - Settings sections

/// The panes available in the Cotypist-style settings window, in sidebar order.
enum SettingsSection: String, CaseIterable, Identifiable {
    case setup
    case general
    case voices
    case shortcuts
    case appSettings
    case pronunciation
    case about

    var id: String { rawValue }

    var label: String {
        switch self {
        case .setup:         return "Setup"
        case .general:       return "General"
        case .voices:        return "Voices"
        case .shortcuts:     return "Shortcuts"
        case .appSettings:   return "App Settings"
        case .pronunciation: return "Pronunciation"
        case .about:         return "About"
        }
    }

    /// Fill color of the rounded-square tile, per the wireframe palette.
    var tileColor: Color {
        switch self {
        case .setup:         return Color(red: 1.00, green: 0.58, blue: 0.00)   // sys-orange
        case .general:       return Color(red: 0.56, green: 0.56, blue: 0.58)   // sys-gray
        case .voices:        return Color(red: 0.00, green: 0.48, blue: 1.00)   // sys-blue
        case .shortcuts:     return Color(red: 0.35, green: 0.34, blue: 0.84)   // sys-indigo
        case .appSettings:   return Color(red: 1.00, green: 0.23, blue: 0.19)   // sys-red
        case .pronunciation: return Color(red: 1.00, green: 0.18, blue: 0.33)   // sys-pink
        case .about:         return Color(red: 0.00, green: 0.48, blue: 1.00)   // sys-blue
        }
    }

    /// SF Symbol shown inside the tile.
    var tileSymbol: String {
        switch self {
        case .setup:         return "lock.fill"
        case .general:       return "gearshape.fill"
        case .voices:        return "waveform"
        case .shortcuts:     return "command"
        case .appSettings:   return "slider.horizontal.3"
        case .pronunciation: return "character.book.closed.fill"
        case .about:         return "info"
        }
    }
}

// MARK: - Colorful rounded-square tile (SidebarTile)

struct SettingsTile: View {
    let color: Color
    let symbol: String

    var body: some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(color)
            .frame(width: 22, height: 22)
            .overlay(
                Image(systemName: symbol)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.white)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.18), lineWidth: 0.5)
            )
            .shadow(color: .black.opacity(0.06), radius: 0.5, y: 0.5)
    }
}

// MARK: - Sidebar item row (SidebarItem)

private struct SettingsSidebarItem: View {
    let section: SettingsSection
    let isSelected: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                SettingsTile(color: section.tileColor, symbol: section.tileSymbol)
                Text(section.label)
                    .font(.system(size: 13))
                    .foregroundStyle(isSelected ? Color.white : Color(nsColor: .labelColor))
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(background)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }

    private var background: Color {
        if isSelected { return Color(red: 0.00, green: 0.48, blue: 1.00) }   // sys-blue
        if hovering { return Color(nsColor: .labelColor).opacity(0.06) }
        return .clear
    }
}

// MARK: - Sidebar (Sidebar)

struct SettingsSidebar: View {
    @Binding var selection: SettingsSection

    /// Sections rendered above the spacer, then those below (About sits at the bottom).
    private let topSections: [SettingsSection] = [
        .setup, .general, .voices, .shortcuts, .appSettings, .pronunciation
    ]
    private let bottomSections: [SettingsSection] = [.about]

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(topSections) { section in
                SettingsSidebarItem(section: section, isSelected: selection == section) {
                    selection = section
                }
            }
            Spacer(minLength: 12)
            ForEach(bottomSections) { section in
                SettingsSidebarItem(section: section, isSelected: selection == section) {
                    selection = section
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 14)
        .frame(width: 200)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}
