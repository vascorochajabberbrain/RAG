#!/usr/bin/env python3
"""
Generate the jB RAG Builder DEV app icon.
Same layout as make_icon.py but with orange accent and "DEV" badge.
Also generates a favicon (32x32 PNG) for the browser tab.

Output: assets/icon_dev_1024.png, assets/favicon_dev.png
Then run: bash make_icns_dev.sh
"""
import struct
import zlib
import os

# ── Canvas ────────────────────────────────────────────────────────────────────
W = H = 1024

# ── Colours (orange accent instead of teal) ──────────────────────────────────
BG        = (0x1a, 0x1a, 0x2e)   # deep navy (same)
CARD      = (0x25, 0x25, 0x50)   # slightly lighter card (same)
ORANGE    = (0xff, 0x6b, 0x00)   # DEV accent orange
WHITE     = (0xff, 0xff, 0xff)
DARK_TEXT = (0x0d, 0x0d, 0x1a)


# ── 5×7 bitmap font ──────────────────────────────────────────────────────────
FONT = {
    'j': [0b00100, 0b00100, 0b00100, 0b00100, 0b10100, 0b10100, 0b01000],
    'B': [0b11100, 0b10010, 0b10010, 0b11100, 0b10010, 0b10010, 0b11100],
    'R': [0b11100, 0b10010, 0b10010, 0b11100, 0b11000, 0b10100, 0b10010],
    'A': [0b01000, 0b10100, 0b10100, 0b11100, 0b10100, 0b10100, 0b10100],
    'G': [0b01110, 0b10000, 0b10000, 0b10110, 0b10001, 0b10001, 0b01110],
    'D': [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
    'E': [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    'V': [0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100],
}


def draw_char(pixels, char, top_x, top_y, scale, color, w, h):
    rows = FONT.get(char)
    if not rows:
        return
    for ry, row in enumerate(rows):
        for cx in range(5):
            if row & (1 << (4 - cx)):
                for dy in range(scale):
                    for dx in range(scale):
                        px = top_x + cx * scale + dx
                        py = top_y + ry * scale + dy
                        if 0 <= px < w and 0 <= py < h:
                            pixels[py][px] = color


def draw_text(pixels, text, cx, cy, scale, color, w, h):
    char_w = 5 * scale
    gap = scale
    total_w = len(text) * char_w + (len(text) - 1) * gap
    start_x = cx - total_w // 2
    char_h = 7 * scale
    start_y = cy - char_h // 2
    for i, ch in enumerate(text):
        draw_char(pixels, ch, start_x + i * (char_w + gap), start_y, scale, color, w, h)


def rounded_rect(pixels, x0, y0, x1, y1, r, color, w, h):
    for y in range(max(0, y0), min(h, y1)):
        for x in range(max(0, x0), min(w, x1)):
            dx = max(x0 + r - x, 0, x - (x1 - r - 1))
            dy = max(y0 + r - y, 0, y - (y1 - r - 1))
            if dx * dx + dy * dy <= r * r:
                pixels[y][x] = color


def filled_rect(pixels, x0, y0, x1, y1, color, w, h):
    for y in range(max(0, y0), min(h, y1)):
        for x in range(max(0, x0), min(w, x1)):
            pixels[y][x] = color


def write_png(path, pixels):
    def make_chunk(tag, data):
        chunk = tag + data
        return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)

    width = len(pixels[0])
    height = len(pixels)
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)

    raw_rows = []
    for row in pixels:
        raw_rows.append(b'\x00')
        for r, g, b in row:
            raw_rows.append(bytes([r, g, b]))
    raw = b''.join(raw_rows)

    idat_data = zlib.compress(raw, 9)

    png = (
        b'\x89PNG\r\n\x1a\n'
        + make_chunk(b'IHDR', ihdr_data)
        + make_chunk(b'IDAT', idat_data)
        + make_chunk(b'IEND', b'')
    )
    with open(path, 'wb') as f:
        f.write(png)
    print(f"  Written: {path} ({os.path.getsize(path):,} bytes)")


def make_app_icon():
    """Generate the 1024x1024 DEV app icon."""
    pixels = [[BG] * W for _ in range(H)]

    # Card
    pad = 40
    rounded_rect(pixels, pad, pad, W - pad, H - pad, r=120, color=CARD, w=W, h=H)

    # Orange accent bar (bottom strip)
    bar_h = 80
    bar_y0 = H - pad - bar_h
    rounded_rect(pixels, pad, bar_y0, W - pad, H - pad, r=0, color=ORANGE, w=W, h=H)
    rounded_rect(pixels, pad, bar_y0, W - pad, H - pad, r=120, color=ORANGE, w=W, h=H)

    # "jB" — large white
    draw_text(pixels, 'jB', W // 2, int(H * 0.33), scale=90, color=WHITE, w=W, h=H)

    # Separator line
    sep_y = int(H * 0.48)
    filled_rect(pixels, W // 2 - 160, sep_y, W // 2 + 160, sep_y + 6, ORANGE, W, H)

    # "RAG" — orange
    draw_text(pixels, 'RAG', W // 2, int(H * 0.57), scale=30, color=ORANGE, w=W, h=H)

    # "DEV" badge — orange rounded rect with white text at bottom
    badge_w = 280
    badge_h = 100
    badge_x = W // 2 - badge_w // 2
    badge_y = int(H * 0.72)
    rounded_rect(pixels, badge_x, badge_y, badge_x + badge_w, badge_y + badge_h, r=20, color=ORANGE, w=W, h=H)
    draw_text(pixels, 'DEV', W // 2, badge_y + badge_h // 2, scale=12, color=DARK_TEXT, w=W, h=H)

    out = os.path.join('assets', 'icon_dev_1024.png')
    print("Generating DEV app icon…")
    write_png(out, pixels)
    return out


def make_favicon():
    """Generate a 32x32 favicon for the browser tab."""
    S = 32
    pixels = [[BG] * S for _ in range(S)]

    # Simple design: orange background with "D" letter
    rounded_rect(pixels, 1, 1, S - 1, S - 1, r=6, color=ORANGE, w=S, h=S)
    draw_text(pixels, 'D', S // 2, int(S * 0.45), scale=3, color=WHITE, w=S, h=S)

    out = os.path.join('assets', 'favicon_dev.png')
    print("Generating DEV favicon…")
    write_png(out, pixels)
    return out


if __name__ == '__main__':
    os.makedirs('assets', exist_ok=True)
    make_app_icon()
    make_favicon()
    print("\nDone. Run: bash make_icns_dev.sh")
