#!/usr/bin/env python3
from __future__ import annotations
"""
cryovision.py — Identify labels in a 10x10 cryogenic freezer box using Claude vision.

Usage:
    python cryovision.py --image box.jpg
    python cryovision.py --image box.jpg --output results.json
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import anthropic
from PIL import Image

ROWS = list("ABCDEFGHIJ")
COLS = [str(n) for n in range(1, 11)]

# Quadrants: (row_slice, col_slice, row_labels, col_labels)
QUADRANTS = [
    ("A–E", "1–5",  ROWS[:5], COLS[:5]),
    ("A–E", "6–10", ROWS[:5], COLS[5:]),
    ("F–J", "1–5",  ROWS[5:], COLS[:5]),
    ("F–J", "6–10", ROWS[5:], COLS[5:]),
]

SYSTEM_PROMPT = """You are a laboratory assistant specializing in cryogenic sample storage.
You have excellent attention to detail and can read small, partially obscured text on tube labels.
Your job is to carefully read images of cryogenic freezer storage box sections and identify
the label on each tube position."""


def make_quadrant_prompt(row_range: str, col_range: str, rows: list, cols: list) -> str:
    positions = [f"{r}{c}" for r in rows for c in cols]
    pos_list = ", ".join(positions)
    return f"""This image shows a SECTION of a 10×10 cryogenic freezer storage box.
This section contains rows {row_range} and columns {col_range}.

Go position by position, left to right, top to bottom. For each tube:
- Look closely at any text printed or written on the cap or side of the tube
- Note any alphanumeric codes, barcodes, or handwritten labels
- Use null if the position is empty or the label is truly unreadable

The positions in this section are: {pos_list}

Respond with ONLY a valid JSON object — no prose, no markdown fences:
{{
  "{positions[0]}": "label or null",
  "{positions[1]}": "label or null",
  ...
  "{positions[-1]}": "label or null"
}}

Include all {len(positions)} positions listed above."""


def encode_pil_image(img: Image.Image, fmt: str = "JPEG") -> tuple[str, str]:
    """Encode a PIL image to base64."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    media_type = "image/jpeg" if fmt == "JPEG" else "image/png"
    return data, media_type


def encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for the given image file."""
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix)
    if media_type is None:
        print(f"Error: unsupported image format '{suffix}'. "
              f"Supported: {', '.join(media_types)}", file=sys.stderr)
        sys.exit(1)

    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


def crop_quadrants(image_path: Path) -> list[tuple[Image.Image, str, str, list, list]]:
    """Split the image into 4 quadrants."""
    img = Image.open(image_path)
    w, h = img.size
    mid_x, mid_y = w // 2, h // 2

    crops = [
        img.crop((0,     0,     mid_x, mid_y)),  # top-left
        img.crop((mid_x, 0,     w,     mid_y)),  # top-right
        img.crop((0,     mid_y, mid_x, h)),      # bottom-left
        img.crop((mid_x, mid_y, w,     h)),      # bottom-right
    ]
    return [(crop, *quad[:-2], quad[2], quad[3])
            for crop, quad in zip(crops, QUADRANTS)]


def parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(line for line in lines if not line.startswith("```")).strip()
    return json.loads(raw)


def analyze_quadrant(
    client: anthropic.Anthropic,
    img: Image.Image,
    row_range: str,
    col_range: str,
    rows: list,
    cols: list,
    quad_num: int,
) -> dict[str, str | None]:
    """Send one quadrant to Claude with thinking enabled."""
    image_data, media_type = encode_pil_image(img)
    prompt = make_quadrant_prompt(row_range, col_range, rows, cols)

    print(f"  Analyzing quadrant {quad_num}/4 (rows {row_range}, cols {col_range})…",
          file=sys.stderr)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    raw = next(
        (block.text for block in response.content if block.type == "text"), ""
    )

    try:
        return parse_response(raw)
    except json.JSONDecodeError as exc:
        print(f"Warning: could not parse quadrant {quad_num} response.\n{exc}",
              file=sys.stderr)
        return {}


def analyze_box(image_path: Path) -> dict[str, str | None]:
    """Split image into quadrants, analyze each with Claude, merge results."""
    client = anthropic.Anthropic()
    grid: dict[str, str | None] = {}

    print("Splitting image into quadrants and analyzing each…", file=sys.stderr)

    quadrant_data = crop_quadrants(image_path)
    for i, (img, row_range, col_range, rows, cols) in enumerate(quadrant_data, 1):
        result = analyze_quadrant(client, img, row_range, col_range, rows, cols, i)
        grid.update(result)

    # Ensure all 100 positions exist
    for r in ROWS:
        for c in COLS:
            grid.setdefault(f"{r}{c}", None)

    return grid


def print_grid(grid: dict[str, str | None]) -> None:
    """Pretty-print the grid as an aligned table."""
    max_label = max(
        (len(str(v)) for v in grid.values() if v is not None),
        default=4,
    )
    cell_width = max(max_label, 4)

    header = "   " + "  ".join(str(c).center(cell_width) for c in COLS)
    separator = "-" * len(header)

    print()
    print(header)
    print(separator)

    for row in ROWS:
        cells = []
        for col in COLS:
            label = grid.get(f"{row}{col}")
            if label is None:
                cells.append("·" * cell_width)
            else:
                cells.append(str(label).center(cell_width)[:cell_width])
        print(f"{row}  " + "  ".join(cells))

    print()

    # Summary counts
    filled = sum(1 for v in grid.values() if v is not None)
    print(f"Filled: {filled}/100   Empty/unread: {100 - filled}/100")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify labels in a 10×10 cryogenic freezer box using Claude vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python cryovision.py --image box.jpg\n"
               "  python cryovision.py --image box.jpg --output results.json",
    )
    parser.add_argument(
        "--image",
        required=True,
        metavar="FILE",
        help="Path to the freezer box image (JPEG, PNG, GIF, or WebP)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Optional path to write JSON output (prints to stdout if omitted)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    grid = analyze_box(image_path)

    json_output = json.dumps(grid, indent=2, sort_keys=True)

    if args.output:
        Path(args.output).write_text(json_output)
        print(f"JSON results written to: {args.output}", file=sys.stderr)
    else:
        print("\n=== JSON Output ===")
        print(json_output)

    print("\n=== Grid View ===")
    print_grid(grid)


if __name__ == "__main__":
    main()
