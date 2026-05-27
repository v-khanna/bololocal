#!/usr/bin/env bash
set -euo pipefail

# Builds a Release .app and packages it into a DMG.
# Output: build/Bolo.dmg
#
# Prerequisites (user must set in project.yml's settings.base.DEVELOPMENT_TEAM
# or via environment):
#   - DEVELOPMENT_TEAM (10-char Apple Developer Team ID, e.g. "AB12CDEF34")
#
# See docs/RELEASE.md for full setup.

cd "$(dirname "$0")/.."

if ! command -v xcodegen >/dev/null 2>&1; then
    echo "Error: xcodegen not installed. Run: brew install xcodegen" >&2
    exit 1
fi

# Read DEVELOPMENT_TEAM from project.yml — fail clearly if blank.
TEAM_ID="$(awk '/DEVELOPMENT_TEAM:/ {gsub(/"/, "", $2); print $2}' project.yml | head -1)"
if [ -z "$TEAM_ID" ] || [ "$TEAM_ID" = '""' ]; then
    cat >&2 <<EOF
Error: DEVELOPMENT_TEAM is empty in project.yml.

Edit project.yml and set:
    settings:
      base:
        DEVELOPMENT_TEAM: "ABCDEF1234"

Find your Team ID at https://developer.apple.com/account → Membership.
See docs/RELEASE.md for full setup.
EOF
    exit 1
fi

BUILD_DIR="build"
ARCHIVE_PATH="$BUILD_DIR/Bolo.xcarchive"
EXPORT_PATH="$BUILD_DIR/Export"
DMG_PATH="$BUILD_DIR/Bolo.dmg"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "→ Regenerating Xcode project from project.yml…"
xcodegen generate

echo "→ Archiving…"
xcodebuild archive \
    -project Bolo.xcodeproj \
    -scheme Bolo \
    -configuration Release \
    -archivePath "$ARCHIVE_PATH" \
    -destination "generic/platform=macOS"

cat > "$BUILD_DIR/ExportOptions.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>developer-id</string>
    <key>signingStyle</key>
    <string>automatic</string>
    <key>teamID</key>
    <string>$TEAM_ID</string>
</dict>
</plist>
EOF

echo "→ Exporting Developer-ID signed .app…"
xcodebuild -exportArchive \
    -archivePath "$ARCHIVE_PATH" \
    -exportPath "$EXPORT_PATH" \
    -exportOptionsPlist "$BUILD_DIR/ExportOptions.plist"

echo "→ Building DMG…"
hdiutil create -volname "Bolo" -srcfolder "$EXPORT_PATH/Bolo.app" -ov -format UDZO "$DMG_PATH"

echo ""
echo "Done. DMG ready at $DMG_PATH"
echo "Run ./scripts/notarize.sh to submit to Apple's notary service."
