#!/usr/bin/env bash
set -euo pipefail

# Notarizes the built DMG via Apple's notarytool, then staples the ticket.
#
# Prerequisite: one-time credential setup via:
#
#     xcrun notarytool store-credentials "hearit-notary" \
#       --apple-id YOUR_APPLE_ID@example.com \
#       --team-id YOUR_TEAM_ID \
#       --password YOUR_APP_SPECIFIC_PASSWORD
#
# See docs/RELEASE.md for App-Specific Password instructions.

cd "$(dirname "$0")/.."

DMG_PATH="build/HearIt.dmg"
PROFILE="hearit-notary"

if [ ! -f "$DMG_PATH" ]; then
    echo "Error: $DMG_PATH not found. Run ./scripts/build.sh first." >&2
    exit 1
fi

if ! xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
    cat >&2 <<EOF
Error: notarytool credentials profile "$PROFILE" not found.

Run this once to store credentials:
    xcrun notarytool store-credentials "$PROFILE" \\
      --apple-id YOUR_APPLE_ID@example.com \\
      --team-id YOUR_TEAM_ID \\
      --password YOUR_APP_SPECIFIC_PASSWORD

See docs/RELEASE.md for full setup.
EOF
    exit 1
fi

echo "→ Submitting $DMG_PATH to Apple notary service…"
xcrun notarytool submit "$DMG_PATH" --keychain-profile "$PROFILE" --wait

echo "→ Stapling notarization ticket…"
xcrun stapler staple "$DMG_PATH"

echo "→ Verifying with spctl…"
spctl --assess --type install --verbose=2 "$DMG_PATH"

echo ""
echo "Done. $DMG_PATH is signed, notarized, and ready for distribution."
