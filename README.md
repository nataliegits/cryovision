# cryovision

Identify sample labels in a 10×10 cryogenic freezer storage box using Claude's vision API.

Point it at a photo of your box and it returns a JSON map of every grid position (A1–J10) with whatever label text it can read.

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
# Print JSON + grid to terminal
python cryovision.py --image box.jpg

# Save JSON to a file (grid still prints to terminal)
python cryovision.py --image box.jpg --output results.json
```

Supported image formats: JPEG, PNG, GIF, WebP.

## Output

**JSON** — a map of all 100 positions. `null` means empty or unreadable.

```json
{
  "A1": "SampleID-001",
  "A2": "SampleID-002",
  "A3": null,
  ...
  "J10": "Control-Neg"
}
```

**Grid view** — an aligned ASCII table printed to the terminal:

```
      1       2       3    ...   10
--------------------------------------------
A  SampleID-001  SampleID-002  ····  ...
B  ····          ····          ····  ...
...
J  ····          ····          ····  Control-Neg

Filled: 87/100   Empty/unread: 13/100
```

## Tips for best results

- Use good, even lighting — shadows across tubes make labels hard to read.
- Shoot straight-on (top-down) to minimise perspective distortion.
- Higher resolution images give Claude more pixel detail per tube.
- If tubes are unlabelled on top, a side-angle shot of the label band may help.

## How it works

1. The image is split into 4 quadrants (top-left, top-right, bottom-left, bottom-right).
2. Each quadrant is sent separately to `claude-opus-4-6` with thinking mode enabled, so Claude gets more pixels per tube and reasons carefully about ambiguous labels.
3. Results from all 4 quadrants are merged into a single 100-position map.
4. Missing positions are filled with `null` and the grid is rendered to the terminal.

---

## Changelog

### v2 — improved accuracy
- **Quadrant splitting:** image is divided into 4 close-up sections before sending to Claude, giving ~4× more pixels per tube
- **Thinking mode:** Claude reasons through ambiguous labels before committing to an answer (`thinking: adaptive`)
- **Better prompting:** Claude now goes position by position and considers text carefully before outputting JSON
- **Summary line:** grid view now shows filled vs empty count at the bottom
- Added `Pillow` dependency for image cropping

### v1 — initial release
- Single-image analysis via Claude vision API
- CLI with `--image` and `--output` flags
- JSON output + ASCII grid view
