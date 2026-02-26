#!/usr/bin/env python3
"""
Generate the jB RAG Builder app icon — pure Python, no external deps.
Uses only struct + zlib (stdlib) to write a valid PNG file.

Output: assets/icon_1024.png
Then run: bash make_icns.sh  (to convert to .icns and wire into the app)
"""
import struct
import zlib
import os
import math

# ── Canvas ────────────────────────────────────────────────────────────────────
W = H = 1024

# ── Colours ───────────────────────────────────────────────────────────────────
BG        = (0x1a, 0x1a, 0x2e)   # deep navy
CARD      = (0x25, 0x25, 0x50)   # slightly lighter card
TEAL      = (0x00, 0xd4, 0xaa)   # accent teal/green
WHITE     = (0xff, 0xff, 0xff)
DARK_TEXT = (0x0d, 0x0d, 0x1a)   # near-black for "RAG" label


# ── 5×7 bitmap font (A-Z, a-z, 0-9) — hand-coded ────────────────────────────
# Each character: 7 rows × 5 cols, stored as 7 ints (bitmask, MSB = leftmost col)
FONT = {
    'j': [0b00100, 0b00100, 0b00100, 0b00100, 0b10100, 0b10100, 0b01000],
    'B': [0b11100, 0b10010, 0b10010, 0b11100, 0b10010, 0b10010, 0b11100],
    'R': [0b11100, 0b10010, 0b10010, 0b11100, 0b11000, 0b10100, 0b10010],
    'A': [0b01000, 0b10100, 0b10100, 0b11100, 0b10100, 0b10100, 0b10100],
    'G': [0b01110, 0b10000, 0b10000, 0b10110, 0b10001, 0b10001, 0b01110],
}


def draw_char(pixels, char, top_x, top_y, scale, color):
    """Draw a single bitmap character scaled up."""
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
                        if 0 <= px < W and 0 <= py < H:
                            pixels[py][px] = color


def draw_text(pixels, text, cx, cy, scale, color):
    """Draw a string centred at (cx, cy)."""
    char_w = 5 * scale
    gap = scale  # 1 pixel gap between chars (scaled)
    total_w = len(text) * char_w + (len(text) - 1) * gap
    start_x = cx - total_w // 2
    char_h = 7 * scale
    start_y = cy - char_h // 2
    for i, ch in enumerate(text):
        draw_char(pixels, ch, start_x + i * (char_w + gap), start_y, scale, color)


def rounded_rect(pixels, x0, y0, x1, y1, r, color):
    """Fill a rounded rectangle."""
    for y in range(y0, y1):
        for x in range(x0, x1):
            # corner check
            dx = max(x0 + r - x, 0, x - (x1 - r - 1))
            dy = max(y0 + r - y, 0, y - (y1 - r - 1))
            if dx * dx + dy * dy <= r * r:
                pixels[y][x] = color


def filled_rect(pixels, x0, y0, x1, y1, color):
    for y in range(y0, y1):
        for x in range(x0, x1):
            pixels[y][x] = color


def write_png(path, pixels):
    """Write an RGB PNG from a list-of-rows of (R,G,B) tuples."""
    def make_chunk(tag, data):
        chunk = tag + data
        return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)

    width = len(pixels[0])
    height = len(pixels)
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)

    raw_rows = []
    for row in pixels:
        raw_rows.append(b'\x00')           # filter type: None
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


def main():
    os.makedirs('assets', exist_ok=True)

    # Initialise canvas
    pixels = [[BG] * W for _ in range(H)]

    # ── Card (rounded rect, inset 40px) ──────────────────────────────────────
    pad = 40
    rounded_rect(pixels, pad, pad, W - pad, H - pad, r=120, color=CARD)

    # ── Teal accent bar (bottom strip inside card) ────────────────────────────
    bar_h = 80
    bar_y0 = H - pad - bar_h
    bar_y1 = H - pad
    # clip corners of bar to match card radius
    rounded_rect(pixels, pad, bar_y0, W - pad, bar_y1, r=0, color=TEAL)
    # restore rounded bottom corners
    rounded_rect(pixels, pad, bar_y0 - 0, W - pad, H - pad, r=120, color=TEAL)

    # ── "jB" — large, white, centred vertically in upper 65% of card ─────────
    jb_scale = 100          # each bitmap pixel → 100px block
    jb_cx = W // 2
    jb_cy = int(H * 0.38)   # upper-centre
    draw_text(pixels, 'jB', jb_cx, jb_cy, scale=jb_scale, color=WHITE)

    # ── "RAG" — smaller, teal-coloured, below jB ─────────────────────────────
    rag_scale = 34
    rag_cy = int(H * 0.64)
    draw_text(pixels, 'RAG', W // 2, rag_cy, scale=rag_scale, color=TEAL)

    # ── Thin separator line between jB and RAG ───────────────────────────────
    sep_y = int(H * 0.535)
    sep_x0 = W // 2 - 160
    sep_x1 = W // 2 + 160
    filled_rect(pixels, sep_x0, sep_y, sep_x1, sep_y + 6, TEAL)

    out = os.path.join('assets', 'icon_1024.png')
    print("Generating icon…")
    write_png(out, pixels)
    print("Done. Open assets/icon_1024.png to preview.")
    print("Next: run  bash make_icns.sh  to build the .icns and install it.")


if __name__ == '__main__':
    main()
