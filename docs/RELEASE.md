# Release Process — HearIt

This document walks through cutting a notarized release suitable for Setapp.

## One-time setup

### 1. Apple Developer Program membership
You need an active paid Apple Developer Program membership (~$99/year) for Developer ID signing. Sign up at https://developer.apple.com/programs/.

### 2. Set your Team ID in project.yml

Find your Team ID at https://developer.apple.com/account → Membership Details.

Edit `project.yml`:

```yaml
settings:
  base:
    DEVELOPMENT_TEAM: "YOUR_TEAM_ID"   # e.g. "AB12CDEF34"
```

Regenerate the Xcode project:

```bash
xcodegen generate
```

### 3. Install a Developer ID Application certificate

In Xcode → Settings → Accounts → Manage Certificates → click `+` → "Developer ID Application". This creates and installs the cert in your login keychain.

### 4. Store notary credentials

You need an [App-Specific Password](https://support.apple.com/en-us/HT204397) for notarytool. Generate one at https://appleid.apple.com → Sign-In and Security → App-Specific Passwords.

Store the credentials once with notarytool (replace with your values):

```bash
xcrun notarytool store-credentials "hearit-notary" \
  --apple-id YOUR_APPLE_ID@example.com \
  --team-id YOUR_TEAM_ID \
  --password YOUR_APP_SPECIFIC_PASSWORD
```

The credentials are stored in your login keychain — never committed to git.

## Cutting a release

After the one-time setup, every release is:

```bash
./scripts/release.sh
```

This runs `build.sh` (archive → sign → DMG) then `notarize.sh` (submit → wait → staple → verify). Total time: 2–10 minutes depending on Apple's notary queue.

The signed, notarized DMG lands at `build/HearIt.dmg`.

## Setapp submission

After your first notarized DMG is ready, follow [docs/SETAPP.md](SETAPP.md) for the submission process.
