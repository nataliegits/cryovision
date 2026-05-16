#!/usr/bin/env python3
from __future__ import annotations
"""
cryovision.py — Identify labels in a 10x10 cryogenic freezer box using Claude vision.

Usage:
    python cryovision.py --image box.jpg
    python cryovision.py --image box.jpg --output results.json
    python cryovision.py --image box.jpg --fast   # skip thinking, use quadrants
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import anthropic
import cv2
import numpy as np
from PIL import Image

ROWS = list("ABCDEFGHIJ")
COLS = [str(n) for n in range(1, 11)]

SYSTEM_PROMPT = """You are a laboratory assistant specializing in cryogenic sample storage.
You have excellent attention to detail and can read small, partially obscured, and handwritten
text on tube caps and labels. Your job is to carefully describe each tube in a freezer box row."""


# ─── OpenCV preprocessing ────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order corner points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def correct_perspective(img: np.ndarray) -> np.ndarray:
    """Detect the box border and warp to a flat top-down view. Falls back gracefully."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0
    min_area = img.shape[0] * img.shape[1] * 0.10  # must cover at least 10% of frame

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > best_area and area > min_area:
            best_quad = approx
            best_area = area

    if best_quad is None:
        return img  # can't detect box outline — return unchanged

    h, w = img.shape[:2]
    pts = order_points(best_quad.reshape(4, 2).astype(np.float32))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (w, h))


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE in LAB space — improves readability of labels under uneven lighting."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Light unsharp mask to make text crisper."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def preprocess(image_path: Path) -> Image.Image:
    """Full OpenCV preprocessing pipeline → PIL Image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)

    img = correct_perspective(img)
    img = enhance_contrast(img)
    img = sharpen(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ─── Image encoding ───────────────────────────────────────────────────────────

def encode_pil(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


def encode_file(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    mt = types.get(suffix)
    if not mt:
        print(f"Error: unsupported format '{suffix}'.", file=sys.stderr)
        sys.exit(1)
    return base64.standard_b64encode(path.read_bytes()).decode(), mt


# ─── Claude analysis ──────────────────────────────────────────────────────────

def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
    return json.loads(raw)


def make_row_prompt(row: str) -> str:
    positions = [f"{row}{c}" for c in COLS]
    return f"""This image is a single horizontal strip showing row {row} of a 10×10
cryogenic freezer storage box. It contains exactly 10 tube positions from left to right:
{", ".join(positions)}.

For each position report — in order of preference:
1. Any printed or handwritten text, ID codes, or gene names on the cap or tube body
2. The cap color (e.g. "red cap", "blue cap") if no text is readable
3. A brief visual note (e.g. "clear tube") if cap color is ambiguous
4. null ONLY if the slot is completely empty — no tube present at all

Respond with ONLY valid JSON, no prose, no markdown:
{{
  "{positions[0]}": "...",
  "{positions[1]}": "...",
  ...
  "{positions[-1]}": "..."
}}

Include all 10 positions."""


def analyze_row(
    client: anthropic.Anthropic,
    strip: Image.Image,
    row: str,
    row_num: int,
    use_thinking: bool,
) -> dict[str, str | None]:
    image_data, media_type = encode_pil(strip)
    prompt = make_row_prompt(row)

    print(f"  Row {row} ({row_num}/10)…", file=sys.stderr)

    kwargs: dict = dict(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media_type, "data": image_data}},
            {"type": "text", "text": prompt},
        ]}],
    )
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    response = client.messages.create(**kwargs)
    raw = next((b.text for b in response.content if b.type == "text"), "")

    try:
        return parse_json(raw)
    except json.JSONDecodeError as exc:
        print(f"  Warning: could not parse row {row} response: {exc}", file=sys.stderr)
        return {f"{row}{c}": None for c in COLS}


def make_quadrant_prompt(row_range: str, col_range: str, rows: list, cols: list) -> str:
    positions = [f"{r}{c}" for r in rows for c in cols]
    return f"""This image shows a SECTION of a 10×10 cryogenic freezer storage box.
This section contains rows {row_range} and columns {col_range}.
Positions (left to right, top to bottom): {", ".join(positions)}.

For each position report cap text, cap color, or a brief description. Use null only for
completely empty slots. Respond with ONLY valid JSON covering all {len(positions)} positions."""


def analyze_quadrant(
    client: anthropic.Anthropic,
    img: Image.Image,
    row_range: str, col_range: str,
    rows: list, cols: list,
    quad_num: int,
) -> dict[str, str | None]:
    image_data, media_type = encode_pil(img)
    prompt = make_quadrant_prompt(row_range, col_range, rows, cols)
    print(f"  Quadrant {quad_num}/4 (rows {row_range}, cols {col_range})…", file=sys.stderr)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media_type, "data": image_data}},
            {"type": "text", "text": prompt},
        ]}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return parse_json(raw)
    except json.JSONDecodeError as exc:
        print(f"  Warning: could not parse quadrant {quad_num}: {exc}", file=sys.stderr)
        return {}


# ─── Main analysis pipeline ───────────────────────────────────────────────────

def analyze_box_rows(image_path: Path, use_thinking: bool) -> dict[str, str | None]:
    """Preprocess with OpenCV, split into row strips, analyze each row."""
    print("Preprocessing image (perspective correction + contrast enhancement)…",
          file=sys.stderr)
    pil_img = preprocess(image_path)
    w, h = pil_img.size
    row_h = h // 10

    client = anthropic.Anthropic()
    grid: dict[str, str | None] = {}

    print(f"Analyzing 10 rows {'with thinking mode' if use_thinking else '(fast mode)'}…",
          file=sys.stderr)
    for i, row in enumerate(ROWS):
        y1 = i * row_h
        y2 = (i + 1) * row_h if i < 9 else h
        strip = pil_img.crop((0, y1, w, y2))
        grid.update(analyze_row(client, strip, row, i + 1, use_thinking))

    for r in ROWS:
        for c in COLS:
            grid.setdefault(f"{r}{c}", None)
    return grid


def analyze_box_quadrants(image_path: Path) -> dict[str, str | None]:
    """Fast mode: 4 quadrant calls, no thinking."""
    pil_img = preprocess(image_path)
    w, h = pil_img.size
    mx, my = w // 2, h // 2

    quadrants = [
        (pil_img.crop((0,  0,  mx, my)), "A–E", "1–5",  ROWS[:5], COLS[:5]),
        (pil_img.crop((mx, 0,  w,  my)), "A–E", "6–10", ROWS[:5], COLS[5:]),
        (pil_img.crop((0,  my, mx, h)),  "F–J", "1–5",  ROWS[5:], COLS[:5]),
        (pil_img.crop((mx, my, w,  h)),  "F–J", "6–10", ROWS[5:], COLS[5:]),
    ]

    client = anthropic.Anthropic()
    grid: dict[str, str | None] = {}
    for i, (img, rr, cr, rows, cols) in enumerate(quadrants, 1):
        grid.update(analyze_quadrant(client, img, rr, cr, rows, cols, i))

    for r in ROWS:
        for c in COLS:
            grid.setdefault(f"{r}{c}", None)
    return grid


# ─── Display ──────────────────────────────────────────────────────────────────

def print_grid(grid: dict[str, str | None]) -> None:
    max_label = max((len(str(v)) for v in grid.values() if v is not None), default=4)
    cell_width = max(max_label, 4)

    header = "   " + "  ".join(str(c).center(cell_width) for c in COLS)
    print()
    print(header)
    print("-" * len(header))
    for row in ROWS:
        cells = []
        for col in COLS:
            label = grid.get(f"{row}{col}")
            cells.append("·" * cell_width if label is None
                         else str(label).center(cell_width)[:cell_width])
        print(f"{row}  " + "  ".join(cells))
    print()
    filled = sum(1 for v in grid.values() if v is not None)
    print(f"Filled: {filled}/100   Empty/unread: {100 - filled}/100")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify labels in a 10×10 cryogenic freezer box using Claude vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cryovision.py --image box.jpg\n"
            "  python cryovision.py --image box.jpg --output results.json\n"
            "  python cryovision.py --image box.jpg --fast"
        ),
    )
    parser.add_argument("--image", required=True, metavar="FILE",
                        help="Path to the freezer box image (JPEG, PNG, GIF, WebP)")
    parser.add_argument("--output", metavar="FILE",
                        help="Write JSON output to this file (default: print to stdout)")
    parser.add_argument("--fast", action="store_true",
                        help="Use 4-quadrant mode without thinking (faster, cheaper)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    if args.fast:
        print("Mode: fast (4 quadrants, no thinking)", file=sys.stderr)
        grid = analyze_box_quadrants(image_path)
    else:
        print("Mode: accurate (10 rows, thinking enabled)", file=sys.stderr)
        grid = analyze_box_rows(image_path, use_thinking=True)

    json_output = json.dumps(grid, indent=2, sort_keys=True)

    if args.output:
        Path(args.output).write_text(json_output)
        print(f"JSON written to: {args.output}", file=sys.stderr)
    else:
        print("\n=== JSON Output ===")
        print(json_output)

    print("\n=== Grid View ===")
    print_grid(grid)


if __name__ == "__main__":
    main()
