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

def cluster_lines(coords: list[int], n_expected: int, img_size: int) -> list[int]:
    """Merge nearby detected lines into cluster centres. Falls back to equal spacing."""
    if len(coords) < 3:
        return [int(img_size * i / n_expected) for i in range(n_expected + 1)]

    coords = sorted(coords)
    threshold = img_size // (n_expected * 3)
    clusters: list[list[int]] = [[coords[0]]]
    for v in coords[1:]:
        if v - clusters[-1][-1] < threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    centres = [int(np.mean(c)) for c in clusters]
    return centres


def detect_grid(img: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Detect the 11×11 grid line positions using Hough lines.
    Returns (y_lines, x_lines) — each a sorted list of 11 pixel positions.
    Falls back to equal division when detection is unreliable.
    """
    h, w = img.shape[:2]

    def equal_split(size: int) -> list[int]:
        return [int(size * i / 10) for i in range(11)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 100)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=int(min(w, h) * 0.25),
        minLineLength=min(w, h) * 0.25,
        maxLineGap=min(w, h) * 0.05,
    )

    if lines is None:
        print("  Grid detection: no lines found, using equal division", file=sys.stderr)
        return equal_split(h), equal_split(w)

    h_coords, v_coords = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 20:
            h_coords.append((y1 + y2) // 2)
        elif angle > 70:
            v_coords.append((x1 + x2) // 2)

    y_centres = cluster_lines(h_coords, 10, h)
    x_centres = cluster_lines(v_coords, 10, w)

    # We need roughly 11 boundary lines. If we found ~10 interior lines, add edges.
    def to_boundaries(centres: list[int], size: int) -> list[int]:
        if not centres:
            return equal_split(size)
        pts = list(centres)
        if pts[0] > size * 0.15:
            pts.insert(0, 0)
        if pts[-1] < size * 0.85:
            pts.append(size)
        # Interpolate to exactly 11
        return [int(v) for v in np.linspace(pts[0], pts[-1], 11)]

    y_lines = to_boundaries(y_centres, h)
    x_lines = to_boundaries(x_centres, w)

    detected = len(y_centres) >= 3 and len(x_centres) >= 3
    method = "Hough lines" if detected else "equal division (fallback)"
    print(f"  Grid detection: {method} "
          f"({len(y_centres)} h-lines, {len(x_centres)} v-lines found)",
          file=sys.stderr)

    return y_lines, x_lines


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
               out_path: Path) -> None:
    """Save a copy of the preprocessed image with the detected grid overlaid."""
    debug = cv_img.copy()
    for y in y_lines:
        cv2.line(debug, (0, y), (debug.shape[1], y), (0, 255, 0), 2)
    for x in x_lines:
        cv2.line(debug, (x, 0), (x, debug.shape[0]), (0, 255, 0), 2)
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
    y_lines, x_lines = detect_grid(cv_img)

    if debug:
        save_debug(cv_img, y_lines, x_lines,
                   image_path.parent / f"{image_path.stem}_debug.jpg")

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
