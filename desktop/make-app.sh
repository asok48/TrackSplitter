#!/bin/sh
# Builds ~/Applications/Encore.app: Electron (Chrome's engine) running desktop/main.js, so Encore has its own window,
# Dock icon, microphone permission and window-only recording (see main.js). Electron is downloaded once into .electron/
# and checked against the SHA-256 sums its release publishes. The app holds only a small stub that loads main.js from
# here, so it needn't be rebuilt when main.js changes. Run by POST /api/encore-app ("Install Encore app"), or by hand.
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
VER=39.1.0
ARCH=x64; [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ] && ARCH=arm64
ZIP="electron-v$VER-darwin-$ARCH.zip"
DIR=".electron/$VER-$ARCH"
URL="https://github.com/electron/electron/releases/download/v$VER"

if [ ! -d "$DIR/Electron.app" ]; then
    mkdir -p "$DIR"
    echo "Downloading Electron $VER ($ARCH)..."
    curl -fL --retry 2 -o "$DIR/$ZIP" "$URL/$ZIP"
    curl -fsL -o "$DIR/SHASUMS256.txt" "$URL/SHASUMS256.txt"
    WANT=$(grep " \*$ZIP\$" "$DIR/SHASUMS256.txt" | cut -d' ' -f1)
    GOT=$(shasum -a 256 "$DIR/$ZIP" | cut -d' ' -f1)
    if [ -z "$WANT" ] || [ "$WANT" != "$GOT" ]; then
        echo "The Electron download didn't match its published checksum; not using it."
        rm -rf "$DIR"
        exit 1
    fi
    ditto -x -k "$DIR/$ZIP" "$DIR"
    rm "$DIR/$ZIP"
fi

APP="$HOME/Applications/Encore.app"
mkdir -p "$HOME/Applications"
if [ -f "$APP/Contents/Resources/app/root.json" ]; then
    rm -rf "$APP"                                      # an earlier build of this app: made here, replaced
elif [ -e "$APP" ]; then                               # anything else (the old launcher): to the Trash
    .venv/bin/python -c 'import sys; from send2trash import send2trash; send2trash(sys.argv[1])' "$APP"
fi
ditto "$DIR/Electron.app" "$APP"

PLIST="$APP/Contents/Info.plist"
set_key() { /usr/libexec/PlistBuddy -c "Set :$1 '$2'" "$PLIST" 2>/dev/null || /usr/libexec/PlistBuddy -c "Add :$1 string '$2'" "$PLIST"; }
set_key CFBundleName Encore
set_key CFBundleDisplayName Encore
set_key CFBundleIdentifier local.track-splitter.encore
set_key NSMicrophoneUsageDescription "Encore listens to your bass through your audio interface."

# Encore's icon (the pick from static/icons/encore-512.png)
ICONSET=$(mktemp -d)/encore.iconset
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
    sips -z $s $s static/icons/encore-512.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2)); [ $d -le 512 ] && sips -z $d $d static/icons/encore-512.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
cp static/icons/encore-512.png "$ICONSET/icon_512x512@2x.png"
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/electron.icns"
rm -rf "$(dirname "$ICONSET")"

# The app itself: a stub loading desktop/main.js from this folder
rm -f "$APP/Contents/Resources/default_app.asar"
mkdir -p "$APP/Contents/Resources/app"
printf '{ "name": "encore", "productName": "Encore", "main": "index.js" }\n' > "$APP/Contents/Resources/app/package.json"
printf "require(require('path').join(require('./root.json').root, 'desktop', 'main.js'));\n" > "$APP/Contents/Resources/app/index.js"
.venv/bin/python -c 'import json, sys; print(json.dumps({"root": sys.argv[1]}))' "$ROOT" > "$APP/Contents/Resources/app/root.json"

# Changing Electron's bundle breaks its signature: sign it again for this Mac (ad hoc), then tell Finder about it
codesign --force --deep --sign - "$APP" >/dev/null 2>&1
touch "$APP"
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
[ -x "$LSREGISTER" ] && "$LSREGISTER" -f "$APP"
echo "$APP"
