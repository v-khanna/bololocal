import SwiftUI

@main
struct HearItApp: App {
    var body: some Scene {
        // Menu bar app: no main window. AppDelegate (added in Task 2)
        // will own the NSStatusItem and popover.
        Settings { EmptyView() }
    }
}
