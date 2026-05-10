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
```

## Tips for best results

- Use good, even lighting — shadows across tubes make labels hard to read.
- Shoot straight-on (top-down) to minimise perspective distortion.
- Higher resolution images give Claude more pixel detail per tube.
- If tubes are unlabelled on top, a side-angle shot of the label band may help.

## How it works

1. The image is base64-encoded and sent to `claude-opus-4-6` via the Anthropic Messages API.
2. Claude is prompted to return a strict JSON object mapping every position to its label.
3. The script parses the response, fills in any missing positions with `null`, and renders both outputs.
