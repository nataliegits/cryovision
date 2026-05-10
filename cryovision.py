#!/usr/bin/env python3
"""
cryovision.py — Identify labels in a 10x10 cryogenic freezer box using Claude vision.

Usage:
    python cryovision.py --image box.jpg
    python cryovision.py --image box.jpg --output results.json
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import anthropic

ROWS = list("ABCDEFGHIJ")
COLS = [str(n) for n in range(1, 11)]

SYSTEM_PROMPT = """You are a laboratory assistant specializing in cryogenic sample storage.
Your job is to read images of 10×10 freezer storage boxes and identify the label or contents
of each position in the grid."""

USER_PROMPT = """This image shows a 10×10 cryogenic freezer storage box.
The grid is labeled with rows A–J (top to bottom) and columns 1–10 (left to right),
giving positions A1 through J10.

Examine the image carefully and return a JSON object mapping every grid position to whatever
label text, tube ID, or content you can read. Use null for positions that appear empty or
whose contents you cannot determine.

Respond with ONLY a valid JSON object — no prose, no markdown fences. Example format:
{
  "A1": "SampleID-001",
  "A2": null,
  ...
  "J10": "Control-Neg"
}

Include all 100 positions (A1–J10)."""


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


def analyze_box(image_path: Path) -> dict[str, str | None]:
    """Send the image to Claude and return the parsed grid map."""
    image_data, media_type = encode_image(image_path)

    client = anthropic.Anthropic()

    print("Sending image to Claude for analysis…", file=sys.stderr)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
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
                    {
                        "type": "text",
                        "text": USER_PROMPT,
                    },
                ],
            }
        ],
    )

    raw = next(
        (block.text for block in response.content if block.type == "text"), ""
    ).strip()

    # Strip accidental markdown fences if Claude adds them despite instructions
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        grid = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Error: Claude returned unparseable JSON.\n{exc}\n\nRaw response:\n{raw}",
              file=sys.stderr)
        sys.exit(1)

    # Ensure all 100 positions exist
    all_positions = {f"{r}{c}" for r in ROWS for c in COLS}
    for pos in all_positions:
        grid.setdefault(pos, None)

    return grid


def print_grid(grid: dict[str, str | None]) -> None:
    """Pretty-print the grid as an aligned table."""
    # Measure max label width for column sizing
    max_label = max(
        (len(str(v)) for v in grid.values() if v is not None),
        default=4,
    )
    cell_width = max(max_label, 4)  # at least 4 chars wide

    header_col_width = 2  # row letter + space

    # Header row (column numbers)
    header = " " * header_col_width + "  ".join(
        str(c).center(cell_width) for c in COLS
    )
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
        print(f"{row} " + "  ".join(cells))

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
