#!/bin/bash
# Build AppIcon.icns for the DEV app from assets/icon_dev_1024.png
# Run after: python3 make_icon_dev.py
set -e

PROJECT="/Users/johanahlund/PROGRAMMING/RAG"
APP="/Users/johanahlund/Applications/jB RAG Builder DEV.app"
SRC="$PROJECT/assets/icon_dev_1024.png"
ICONSET="$PROJECT/assets/AppIconDev.iconset"
ICNS="$PROJECT/assets/AppIconDev.icns"

echo "==> Building DEV .iconset from $SRC"
mkdir -p "$ICONSET"

for SIZE in 16 32 64 128 256 512; do
    sips -z $SIZE $SIZE "$SRC" --out "$ICONSET/icon_${SIZE}x${SIZE}.png"       > /dev/null
    sips -z $((SIZE*2)) $((SIZE*2)) "$SRC" --out "$ICONSET/icon_${SIZE}x${SIZE}@2x.png" > /dev/null
    echo "  icon_${SIZE}x${SIZE}  +  @2x"
done

echo "==> Running iconutil"
iconutil -c icns "$ICONSET" -o "$ICNS"
echo "  Created: $ICNS ($(du -sh "$ICNS" | cut -f1))"

echo "==> Installing icon into DEV .app bundle"
RESOURCES="$APP/Contents/Resources"
mkdir -p "$RESOURCES"
cp "$ICNS" "$RESOURCES/AppIcon.icns"

echo "==> Refreshing Launch Services + Dock"
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$APP" 2>/dev/null || true

killall Dock 2>/dev/null || true

echo ""
echo "✅ Done! DEV icon installed."
