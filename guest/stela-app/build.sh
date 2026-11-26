#!/usr/bin/env bash
# Build a standalone app with the canonical branding; no Rust or EFI build is required.
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
    echo "Stela.app packaging requires macOS and Xcode command line tools." >&2
    exit 1
fi

package="$(cd -- "$(dirname -- "$0")" && pwd)"
repository="$(cd -- "$package/../.." && pwd)"
architecture="${STELA_APP_ARCH:-arm64}"
case "$architecture" in arm64|x86_64) ;; *) echo "STELA_APP_ARCH must be arm64 or x86_64" >&2; exit 1 ;; esac

swift test --package-path "$package"
swift build --package-path "$package" --configuration release --arch "$architecture" --product Stela
binary_directory="$(swift build --package-path "$package" --configuration release --arch "$architecture" --show-bin-path)"
output="$repository/target/stela-app"
mkdir -p "$output"
stage="$(mktemp -d "$output/package.XXXXXX")"
app="$stage/Stela.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$binary_directory/Stela" "$app/Contents/MacOS/Stela"
cp "$package/Packaging/Info.plist" "$app/Contents/Info.plist"
cp "$repository/assets/northwood-nw.png" "$app/Contents/Resources/stela-icon.png"

# Use the workspace package version rather than maintaining a second product version.
version="$(awk '/^\[workspace.package\]/{section=1;next} section && /^\[/{exit} section && /^version = /{gsub(/"/, "", $3);print $3;exit}' "$repository/Cargo.toml")"
[[ -n "$version" ]] || { echo "Cannot read the workspace version" >&2; exit 1; }
# A copy of the sources without Git (a Mac or guest build) passes the revision it came from.
if [[ -n "${STELA_SOURCE_REVISION:-}" ]]; then
    revision="$STELA_SOURCE_REVISION"
elif git -C "$repository" rev-parse --git-dir >/dev/null 2>&1; then
    revision="$(git -C "$repository" rev-parse --short=12 HEAD)"
    if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=normal -- guest/stela-app assets Cargo.toml)" ]]; then
        revision="$revision-modified"
    fi
else
    echo "No Git checkout: set STELA_SOURCE_REVISION to the sources' revision" >&2
    exit 1
fi
[[ "$revision" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "STELA_SOURCE_REVISION must be letters, digits, dots and dashes" >&2; exit 1; }
short_version="${version%%+*}"
short_version="${short_version%%-*}"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $short_version" "$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :StelaSourceRevision $revision" "$app/Contents/Info.plist"

iconset="$stage/Stela.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$repository/assets/northwood-nw.png" --out "$iconset/icon_${size}x${size}.png" >/dev/null
    doubled=$((size * 2))
    sips -z "$doubled" "$doubled" "$repository/assets/northwood-nw.png" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil --convert icns "$iconset" --output "$app/Contents/Resources/Stela.icns"
plutil -lint "$app/Contents/Info.plist"

# Local and debug symbols only enlarge the copy stela.efi carries into the guest.
strip -S -x "$app/Contents/MacOS/Stela"

identity="${STELA_SIGN_IDENTITY:--}"
if [[ "$identity" == - ]]; then
    codesign --force --sign - "$app"
else
    codesign --force --options runtime --timestamp --sign "$identity" "$app"
fi
codesign --verify --strict "$app"

# Each build is a fresh bundle. Print its exact path, without replacing an installed app.
printf '\nBuilt %s (%s):\n%s\n' "$version" "$architecture" "$app"
