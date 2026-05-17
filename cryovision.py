#!/usr/bin/env python3
from __future__ import annotations
"""
cryovision.py — Identify labels in a 10x10 cryogenic freezer box using Claude vision.

Usage:
    python cryovision.py --image box.jpg
    python cryovision.py --image box.jpg --output results.json
    python cryovision.py --image box.jpg --output results.csv
    python cryovision.py --image box.jpg --fast   # row strips, no thinking
    python cryovision.py --image box.jpg --debug  # save preprocessed image + grid overlay
"""

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

import anthropic
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROWS = list("ABCDEFGHIJ")
COLS = [str(n) for n in range(1, 11)]

CELL_SIZE = 160   # px — each cell is resized to this before compositing
CELL_GAP  = 4     # px — gap between cells in composite
LABEL_H   = 22    # px — height of column-number label bar

SYSTEM_PROMPT = """You are a laboratory assistant specializing in cryogenic sample storage.
You have excellent attention to detail and can read small, partially obscured, and handwritten
text on tube caps and labels."""


# ─── OpenCV preprocessing ────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def correct_perspective(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad, best_area = None, 0
    min_area = img.shape[0] * img.shape[1] * 0.10

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > best_area and area > min_area:
            best_quad, best_area = approx, area

    if best_quad is None:
        return img

    h, w = img.shape[:2]
    pts = order_points(best_quad.reshape(4, 2).astype(np.float32))
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w, h))


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def sharpen(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def preprocess_cv(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)
    img = correct_perspective(img)
    img = enhance_contrast(img)
    img = sharpen(img)
    return img


# ─── Grid detection ───────────────────────────────────────────────────────────

def equal_split(size: int) -> list[int]:
    return [int(size * i / 10) for i in range(11)]


def centers_to_boundaries(centers: np.ndarray, size: int) -> list[int]:
    """Convert 10 cell centres to 11 boundary lines."""
    centers = np.sort(centers)
    spacing = np.median(np.diff(centers)) if len(centers) > 1 else size / 10
    boundaries = [int(centers[0] - spacing / 2)]
    for i in range(len(centers) - 1):
        boundaries.append(int((centers[i] + centers[i + 1]) / 2))
    boundaries.append(int(centers[-1] + spacing / 2))
    # Clamp to image
    boundaries = [max(0, min(size, b)) for b in boundaries]
    # Interpolate to exactly 11 points
    return [int(v) for v in np.linspace(boundaries[0], boundaries[-1], 11)]


def cluster_1d(coords: np.ndarray, n: int) -> np.ndarray:
    """
    Find n cluster centres in a 1-D array using percentile binning.
    More robust than linspace(min, max) because it uses actual circle positions
    and is not thrown off by a single outlier circle near the edge.
    """
    coords = np.sort(coords)
    centers = []
    for i in range(n):
        lo = np.percentile(coords, 100 * i / n)
        hi = np.percentile(coords, 100 * (i + 1) / n)
        bucket = coords[(coords >= lo) & (coords <= hi)]
        if len(bucket):
            centers.append(float(np.median(bucket)))
    return np.array(centers)


def detect_grid(img: np.ndarray) -> tuple[list[int], list[int], object]:
    """
    Detect tube caps as circles with HoughCircles, fit a 10×10 grid to their
    centres. Falls back to equal division if too few circles are found.
    Returns (y_lines, x_lines, circles_or_None).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    short = min(h, w)
    # Each tube cap is ~1/10 of the box; radius is ~1/20.
    # minDist ≈ one tube diameter prevents double-detecting the inner/outer cap rings.
    min_r    = int(short / 22)
    max_r    = int(short / 13)
    min_dist = int(short / 10)   # ~1 tube-width apart — no two circles on the same tube

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=min_dist,
        param1=80, param2=50,    # stricter: fewer false positives
        minRadius=min_r, maxRadius=max_r,
    )

    if circles is None or len(circles[0]) < 20:
        n = 0 if circles is None else len(circles[0])
        print(f"  Grid detection: found {n} circles — using equal division", file=sys.stderr)
        return equal_split(h), equal_split(w), None

    # Drop circles too close to the image border (likely box frame, not tubes)
    margin = short * 0.04
    mask = (
        (circles[0, :, 0] > margin) & (circles[0, :, 0] < w - margin) &
        (circles[0, :, 1] > margin) & (circles[0, :, 1] < h - margin)
    )
    kept = circles[0][mask]

    if len(kept) < 20:
        print(f"  Grid detection: only {len(kept)} circles after edge filter — using equal division",
              file=sys.stderr)
        return equal_split(h), equal_split(w), circles  # still show raw circles in debug

    cx, cy = kept[:, 0], kept[:, 1]
    print(f"  Grid detection: found {len(cx)} tube circles", file=sys.stderr)

    col_centers = cluster_1d(cx, 10)
    row_centers = cluster_1d(cy, 10)

    if len(col_centers) < 10 or len(row_centers) < 10:
        print("  Grid detection: could not resolve 10 columns/rows — using equal division",
              file=sys.stderr)
        return equal_split(h), equal_split(w), circles

    x_lines = centers_to_boundaries(col_centers, w)
    y_lines = centers_to_boundaries(row_centers, h)
    return y_lines, x_lines, circles


# ─── Cell cropping & compositing ─────────────────────────────────────────────

def crop_cells(pil_img: Image.Image,
               y_lines: list[int],
               x_lines: list[int]) -> list[list[Image.Image]]:
    """Return a 10×10 list of cropped cell images."""
    cells = []
    for i in range(10):
        row_cells = []
        for j in range(10):
            cell = pil_img.crop((x_lines[j], y_lines[i], x_lines[j+1], y_lines[i+1]))
            row_cells.append(cell)
        cells.append(row_cells)
    return cells


def make_row_composite(cells: list[Image.Image], row_label: str) -> Image.Image:
    """
    Tile 10 cell crops into one labelled composite image.
    Column numbers (1-10) are printed above each cell so Claude can orient itself.
    """
    total_w = CELL_SIZE * 10 + CELL_GAP * 9
    total_h = CELL_SIZE + LABEL_H
    composite = Image.new("RGB", (total_w, total_h), (50, 50, 50))
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    for j, cell in enumerate(cells):
        x_off = j * (CELL_SIZE + CELL_GAP)

        # Column label
        col_label = str(j + 1)
        draw.text((x_off + CELL_SIZE // 2, 3), col_label,
                  fill=(220, 220, 220), font=font, anchor="mt")

        # Cell image
        resized = cell.resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
        composite.paste(resized, (x_off, LABEL_H))

    return composite


def save_debug(cv_img: np.ndarray,
               y_lines: list[int],
               x_lines: list[int],
               out_path: Path,
               circles: np.ndarray | None = None) -> None:
    """Save preprocessed image with detected grid and circles overlaid."""
    debug = cv_img.copy()
    # Grid lines in green
    for y in y_lines:
        cv2.line(debug, (0, y), (debug.shape[1], y), (0, 255, 0), 2)
    for x in x_lines:
        cv2.line(debug, (x, 0), (x, debug.shape[0]), (0, 255, 0), 2)
    # Detected circles in cyan
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            cv2.circle(debug, (cx, cy), r, (255, 255, 0), 2)
            cv2.circle(debug, (cx, cy), 3, (255, 255, 0), -1)
    cv2.imwrite(str(out_path), debug)
    print(f"  Debug image saved: {out_path}", file=sys.stderr)


# ─── Image encoding ───────────────────────────────────────────────────────────

def encode_pil(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


# ─── Claude analysis ──────────────────────────────────────────────────────────

def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
    return json.loads(raw)


def make_cell_prompt(row: str) -> str:
    positions = [f"{row}{c}" for c in COLS]
    return f"""This image shows row {row} of a 10×10 cryogenic freezer storage box.
It contains 10 individual tube crops arranged left to right, labeled 1–10 at the top.
Each crop is a single tube cap viewed from above.

The positions are: {", ".join(positions)} (left to right).

For each position report — in order of preference:
1. Any printed or handwritten text, ID codes, or gene names on the cap or tube
2. The cap color (e.g. "red cap", "blue cap") if no text is readable
3. A brief visual note (e.g. "clear tube") if needed
4. null ONLY if the slot is completely empty — no tube present

Respond with ONLY valid JSON, no prose, no markdown:
{{
  "{positions[0]}": "...",
  ...
  "{positions[-1]}": "..."
}}

Include all 10 positions."""


def make_row_strip_prompt(row: str) -> str:
    positions = [f"{row}{c}" for c in COLS]
    return f"""This image is row {row} of a 10×10 cryogenic freezer storage box —
a horizontal strip with 10 tubes from left to right: {", ".join(positions)}.

Report cap text, cap color, or a brief description for each. Use null only for empty slots.
Respond with ONLY valid JSON covering all 10 positions."""


def call_claude(
    client: anthropic.Anthropic,
    img: Image.Image,
    prompt: str,
    use_thinking: bool,
    row: str,
) -> dict[str, str | None]:
    image_data, media_type = encode_pil(img)
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
        print(f"  Warning: could not parse row {row}: {exc}", file=sys.stderr)
        return {f"{row}{c}": None for c in COLS}


# ─── Main pipelines ───────────────────────────────────────────────────────────

def analyze_box_cells(image_path: Path, debug: bool) -> dict[str, str | None]:
    """
    Full pipeline:
      1. OpenCV preprocessing
      2. Hough-line grid detection (fallback: equal division)
      3. Crop 100 individual cells
      4. Composite each row into a labelled image
      5. Send each row to Claude with thinking enabled
    """
    print("Preprocessing image…", file=sys.stderr)
    cv_img = preprocess_cv(image_path)
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    print("Detecting grid…", file=sys.stderr)
    y_lines, x_lines, circles = detect_grid(cv_img)

    if debug:
        save_debug(cv_img, y_lines, x_lines,
                   image_path.parent / f"{image_path.stem}_debug.jpg",
                   circles=circles)

    cells = crop_cells(pil_img, y_lines, x_lines)

    client = anthropic.Anthropic()
    grid: dict[str, str | None] = {}

    print("Analyzing rows (cell crops, thinking enabled)…", file=sys.stderr)
    for i, row in enumerate(ROWS):
        print(f"  Row {row} ({i+1}/10)…", file=sys.stderr)
        composite = make_row_composite(cells[i], row)
        prompt = make_cell_prompt(row)
        grid.update(call_claude(client, composite, prompt, use_thinking=True, row=row))

    for r in ROWS:
        for c in COLS:
            grid.setdefault(f"{r}{c}", None)
    return grid


def analyze_box_fast(image_path: Path) -> dict[str, str | None]:
    """Fast mode: row strips, no thinking."""
    print("Preprocessing image…", file=sys.stderr)
    cv_img = preprocess_cv(image_path)
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    w, h = pil_img.size
    row_h = h // 10

    client = anthropic.Anthropic()
    grid: dict[str, str | None] = {}

    print("Analyzing rows (strip mode, no thinking)…", file=sys.stderr)
    for i, row in enumerate(ROWS):
        print(f"  Row {row} ({i+1}/10)…", file=sys.stderr)
        y1 = i * row_h
        y2 = (i + 1) * row_h if i < 9 else h
        strip = pil_img.crop((0, y1, w, y2))
        grid.update(call_claude(client, strip, make_row_strip_prompt(row),
                                use_thinking=False, row=row))

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
            "  python cryovision.py --image box.jpg --output results.csv\n"
            "  python cryovision.py --image box.jpg --fast\n"
            "  python cryovision.py --image box.jpg --debug"
        ),
    )
    parser.add_argument("--image", required=True, metavar="FILE",
                        help="Path to the freezer box image (JPEG, PNG, GIF, WebP)")
    parser.add_argument("--output", metavar="FILE",
                        help="Save results to .json or .csv (default: print to stdout)")
    parser.add_argument("--fast", action="store_true",
                        help="Row-strip mode, no thinking — faster and cheaper")
    parser.add_argument("--debug", action="store_true",
                        help="Save a debug image showing the detected grid overlay")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    if args.fast:
        print("Mode: fast (row strips, no thinking)", file=sys.stderr)
        grid = analyze_box_fast(image_path)
    else:
        print("Mode: accurate (cell crops + grid detection + thinking)", file=sys.stderr)
        grid = analyze_box_cells(image_path, debug=args.debug)

    if args.output:
        out = Path(args.output)
        if out.suffix.lower() == ".csv":
            with out.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["position", "row", "column", "label"])
                for r in ROWS:
                    for c in COLS:
                        pos = f"{r}{c}"
                        writer.writerow([pos, r, c, grid.get(pos) or ""])
            print(f"CSV written to: {out}", file=sys.stderr)
        else:
            out.write_text(json.dumps(grid, indent=2, sort_keys=True))
            print(f"JSON written to: {out}", file=sys.stderr)
    else:
        print("\n=== JSON Output ===")
        print(json.dumps(grid, indent=2, sort_keys=True))

    print("\n=== Grid View ===")
    print_grid(grid)


if __name__ == "__main__":
    main()
