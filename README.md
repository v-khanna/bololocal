# HearIt (working name)

macOS menu bar app that reads selected text aloud in a natural AI voice.
Fully on-device. Apple Silicon only. macOS 15+.

See [SCOPE.md](SCOPE.md) for product scope.
See [docs/superpowers/plans/](docs/superpowers/plans/) for the implementation plan.

## Build

```bash
xcodegen generate                  # regenerate HearIt.xcodeproj from project.yml
xcodebuild -scheme HearIt build    # build
xcodebuild -scheme HearIt test     # run tests
```

> **Note:** Builds require `DEVELOPMENT_TEAM` to be filled in `project.yml` (under `settings.base`).
> For unsigned local builds, pass `CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO` to xcodebuild.

## Release

To cut a signed + notarized DMG for Setapp distribution, see [docs/RELEASE.md](docs/RELEASE.md).
