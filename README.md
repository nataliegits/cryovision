# cryovision 🧊📦

Ever spent 20 minutes digging through a freezer box looking for one tube? Same. CryoVision uses OpenCV and Claude's vision API to read your 10×10 cryobox from a single photo and exports the info so you can have it on hand. No more guessing which box has "Primer 132689."

Point it at a photo of your box and it returns a JSON map of every grid position (A1–J10) with whatever label text, cap color, or description it can read. Save that map for your own reference.

## Requirements

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

```bash
# Accurate mode: grid detection + individual cell crops + thinking (default)
python cryovision.py --image box.jpg

# Fast mode: row strips, no thinking — quicker and cheaper
python cryovision.py --image box.jpg --fast

# Save as JSON
python cryovision.py --image box.jpg --output results.json

# Save as CSV (opens in Excel / Google Sheets)
python cryovision.py --image box.jpg --output results.csv

# Debug: save a copy of the image with the detected grid overlaid
python cryovision.py --image box.jpg --debug
```

Supported image formats: JPEG, PNG, GIF, WebP.

## Output

**JSON** — a map of all 100 positions. `null` means the slot is empty.

```json
{
  "A1": "SampleID-001",
  "A2": "red cap",
  "A3": null,
  ...
  "J10": "GAPDH-F"
}
```

**CSV** — four columns: `position`, `row`, `column`, `label`. Easy to open in Excel or Google Sheets and filter/sort by row or label.

```
position,row,column,label
A1,A,1,SampleID-001
A2,A,2,red cap
A3,A,3,
...
J10,J,10,GAPDH-F
```

**Grid view** — an aligned ASCII table with a filled/empty summary:

```
      1         2         3      ...    10
--------------------------------------------------
A  SampleID-001  red cap   ·····  ...
B  ·····         ·····     ·····  ...
...
J  ·····         ·····     ·····  GAPDH-F

Filled: 87/100   Empty/unread: 13/100
```

## How it works

1. **OpenCV preprocessing** — perspective correction (detects the box border and flattens tilt), CLAHE contrast enhancement, and sharpening to make labels more legible
2. **Circle-based grid detection** — HoughCircles detects tube caps as circles and fits the 10×10 grid to where the tubes actually are; falls back to equal division if too few circles are found
3. **Row compositing** — the 10 cell crops per row are tiled into a single labelled image with column numbers above each cell
4. **Claude vision** — each row composite is sent to `claude-opus-4-6` with thinking mode enabled; Claude sees a close-up of each individual tube cap and reasons carefully before answering
5. **Graceful fallback** — if a row can't be parsed, it's filled with `null` rather than crashing

Use `--debug` to save a copy of the preprocessed image with two overlays:
- **Green lines** — the 10×10 grid boundaries used to crop each cell
- **Cyan circles** — the individual tube caps detected by HoughCircles

Check this image before running a full analysis — if the circles are landing on the caps and the green lines sit between tubes, detection is working correctly. If not, it tells you exactly what's off.

## Tips for best results

- Shoot straight-on (top-down) — perspective correction helps with mild tilt but can't fix extreme angles
- Use even lighting — shadows across caps are the main cause of misreads
- Higher resolution is better — more pixels per cap means more readable text
- For handwritten gene names on caps: good lighting + a close shot beats any amount of prompting

---

## Build log

### Background

Built for a molecular biology lab that stores DNA oligos and PCR primers in 10×10 cryoboxes. The problem: keeping track of which tube is where when caps are labeled by hand (gene names, primer IDs, etc.) and the box layout changes over time.

Initial approach: use Claude's vision API as a quick wrapper — no training data needed, works out of the box.

### v6 — HoughCircles tuning: one circle per tube

The debug overlay showed cyan circles stacking on top of each other — two or three rings detected per tube instead of one. The cause: every cryotube cap has at least two visually distinct circular features: the raised outer screw rim and the flat inner surface. With the original parameters, HoughCircles was finding both rings on every tube.

Three parameter changes fixed it:

- **`minDist` raised from `short/14` → `short/10`** — this is the minimum pixel distance allowed between any two detected circles. Raising it to roughly one full tube-width means the algorithm physically cannot place a second circle on the same tube it already found.
- **`param2` raised from 30 → 50** — this is the accumulator threshold: how much evidence OpenCV requires before it commits to a circle. Higher = stricter = fewer false positives from cap ridges and reflections.
- **`param1` raised from 50 → 80** — the upper Canny edge threshold used internally. Higher = only the sharpest edges (cap rims) count toward circle evidence; softer gradients (label text, shadows) are ignored.

On top of that, the grid-fitting logic was upgraded from `np.linspace(min, max, 10)` to percentile-bin clustering: the detected circle centres are sorted and bucketed into 10 equal-percentile bins, and the median of each bin becomes the column or row centre. This means one outlier circle near the box edge can't drag the entire grid off — it just lands in a bin with a few other outliers and gets median'd away.

### v5 — circle-based grid detection + individual cell crops
- **Circle-based grid detection** — OpenCV detects tube caps as circles (HoughCircles) and fits the 10×10 grid to where the tubes actually are, rather than assuming equal spacing; falls back to equal division if too few circles are found
- **Individual cell crops** — each of the 100 tube positions is cropped individually and composited into a labelled row image before being sent to Claude; Claude now sees one tube at a time rather than a strip of 10
- **`--debug` flag** — saves the preprocessed image with green grid lines and cyan circles overlaid so you can verify detection before spending API credits
- **`--fast` mode** — row strips without thinking, for quicker cheaper runs

### v4 — CSV export
- `--output results.csv` exports a spreadsheet-friendly CSV with columns `position`, `row`, `column`, `label`
- `--output results.json` still works — format is auto-detected from the file extension

### v3 — OpenCV preprocessing + row-by-row analysis
*Motivation: shared with the lab group; feedback was that handwritten cap labels weren't parsing accurately. Group recommended pivoting to traditional CV (OpenCV, YOLO, SAM, Roboflow).*

- **OpenCV pipeline** added before Claude: perspective warp, CLAHE contrast enhancement, unsharp mask sharpening
- **Row-by-row splitting** replaces quadrants — Claude now sees a 1×10 strip per call (10 calls) instead of a 5×5 quadrant (4 calls), giving ~2.5× more pixels per tube
- **`--fast` flag** to fall back to quadrant mode when speed/cost matters

### v2 — improved accuracy
- **Quadrant splitting** — image divided into 4 sections before sending to Claude
- **Thinking mode** — Claude reasons through ambiguous labels before committing (`thinking: adaptive`)
- **Better prompting** — position-by-position instructions, cap color as fallback
- **Summary line** — filled vs empty count at the bottom of the grid

### v1 — initial release
- Single-image analysis via Claude vision API
- CLI with `--image` and `--output` flags
- JSON output + ASCII grid view

---

## Roadmap

The group recommended moving toward traditional computer vision for better accuracy on handwritten labels. Planned next steps:

- [x] **Individual tube crops** — detect tube caps as circles and crop each of the 100 cells individually before sending to Claude
- [x] **Circle-based grid detection** — fit the grid to detected tube centres rather than assuming equal spacing
- [ ] **YOLO tube detection** — train a model to locate tubes regardless of box orientation or partial occlusion
- [ ] **Roboflow training pipeline** — label a dataset of freezer box images for fine-tuning
- [ ] **Replace Claude with local OCR** — once tube positions are reliably detected, run Tesseract or a fine-tuned text recognition model on each crop for offline, zero-cost operation
