# cryovision

Identify sample labels in a 10×10 cryogenic freezer storage box using Claude's vision API.

Point it at a photo of your box and it returns a JSON map of every grid position (A1–J10) with whatever label text, cap color, or description it can read.

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
# Accurate mode: OpenCV preprocessing + row-by-row analysis with thinking (default)
python cryovision.py --image box.jpg

# Fast mode: 4 quadrants, no thinking — quicker and cheaper
python cryovision.py --image box.jpg --fast

# Save as JSON
python cryovision.py --image box.jpg --output results.json

# Save as CSV (opens in Excel / Google Sheets)
python cryovision.py --image box.jpg --output results.csv
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
2. **Row-by-row splitting** — the image is cut into 10 horizontal strips, one per row (A–J)
3. **Claude vision** — each strip is sent to `claude-opus-4-6` with thinking mode enabled, so Claude sees a tight close-up of 10 tubes at a time and reasons carefully about each one
4. **Graceful fallback** — if a row can't be parsed, it's filled with `null` rather than crashing

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

### v3 — OpenCV preprocessing + row-by-row analysis
*Motivation: shared with the lab group; feedback was that handwritten cap labels weren't parsing accurately. Group recommended pivoting to traditional CV (OpenCV, YOLO, SAM, Roboflow).*

Implemented a hybrid approach as a stepping stone:
- **OpenCV pipeline** added before Claude: perspective warp, CLAHE contrast enhancement, unsharp mask sharpening
- **Row-by-row splitting** replaces quadrants — Claude now sees a 1×10 strip per call (10 calls) instead of a 5×5 quadrant (4 calls), giving ~2.5× more pixels per tube
- **`--fast` flag** to fall back to quadrant mode when speed/cost matters
- Roadmap: move toward YOLO tube detection + individual cell crops → eventually replace Claude with a trained OCR model

### v4 — CSV export
- `--output results.csv` now exports a spreadsheet-friendly CSV with columns `position`, `row`, `column`, `label`
- `--output results.json` still works as before — format is auto-detected from the file extension

### v2 — improved accuracy
- **Quadrant splitting:** image divided into 4 sections before sending to Claude
- **Thinking mode:** Claude reasons through ambiguous labels before committing (`thinking: adaptive`)
- **Better prompting:** position-by-position instructions, cap color as fallback
- **Summary line:** filled vs empty count at the bottom of the grid

### v1 — initial release
- Single-image analysis via Claude vision API
- CLI with `--image` and `--output` flags
- JSON output + ASCII grid view

---

## Roadmap

The group recommended moving toward traditional computer vision for better accuracy on handwritten labels. Planned next steps:

- [ ] **Individual tube crops** — detect grid lines with OpenCV and crop each of the 100 cells individually before sending to Claude
- [ ] **YOLO tube detection** — train a model to locate tubes regardless of box orientation or partial occlusion
- [ ] **Roboflow training pipeline** — label a dataset of freezer box images for fine-tuning
- [ ] **Replace Claude with local OCR** — once tube positions are reliably detected, run Tesseract or a fine-tuned text recognition model on each crop for offline, zero-cost operation
