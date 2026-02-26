#!/bin/bash
# Build AppIcon.icns from assets/icon_1024.png and install into the .app bundle.
# Run after: python make_icon.py
set -e

PROJECT="/Users/johanahlund/PROGRAMMING/RAG"
APP="/Users/johanahlund/Applications/jB RAG Builder.app"
SRC="$PROJECT/assets/icon_1024.png"
ICONSET="$PROJECT/assets/AppIcon.iconset"
ICNS="$PROJECT/assets/AppIcon.icns"

echo "==> Building .iconset from $SRC"
mkdir -p "$ICONSET"

for SIZE in 16 32 64 128 256 512; do
    sips -z $SIZE $SIZE "$SRC" --out "$ICONSET/icon_${SIZE}x${SIZE}.png"       > /dev/null
    sips -z $((SIZE*2)) $((SIZE*2)) "$SRC" --out "$ICONSET/icon_${SIZE}x${SIZE}@2x.png" > /dev/null
    echo "  icon_${SIZE}x${SIZE}  +  @2x"
done

echo "==> Running iconutil"
iconutil -c icns "$ICONSET" -o "$ICNS"
echo "  Created: $ICNS ($(du -sh "$ICNS" | cut -f1))"

echo "==> Installing icon into .app bundle"
RESOURCES="$APP/Contents/Resources"
mkdir -p "$RESOURCES"
cp "$ICNS" "$RESOURCES/AppIcon.icns"

echo "==> Updating Info.plist"
PLIST="$APP/Contents/Info.plist"
# Add CFBundleIconFile if not already present
if ! grep -q "CFBundleIconFile" "$PLIST"; then
    # Insert before closing </dict>
    sed -i '' 's|</dict>|    <key>CFBundleIconFile</key>\
    <string>AppIcon</string>\
</dict>|' "$PLIST"
    echo "  Added CFBundleIconFile to Info.plist"
else
    echo "  CFBundleIconFile already in Info.plist"
fi

echo "==> Refreshing Launch Services + Dock"
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$APP" 2>/dev/null || true

# Restart Dock to flush icon cache
killall Dock 2>/dev/null || true

echo ""
echo "✅ Done! Icon installed. Check your Dock in a moment."
echo "   If the icon still looks blank, right-click the Dock icon → Remove,"
echo "   then drag ~/Applications/jB RAG Builder.app back to the Dock."
