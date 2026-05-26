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
