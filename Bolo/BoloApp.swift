import SwiftUI

@main
struct BoloApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        SwiftUI.Settings { EmptyView() }
    }
}
