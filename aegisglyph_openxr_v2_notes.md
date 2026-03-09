# AegisGlyph OpenXR v2 — Notes

## What this file is
A Python v2 architecture for an accessibility-oriented XR guidance engine.

## What it does today
- Optional OpenXR capability probing through `pyopenxr` if installed
- Detects common OpenXR extensions such as hand tracking and eye gaze
- Runs a deterministic simulator when no runtime or bindings are available
- Produces symbolic guidance frames and JSONL logs
- Filters noisy / contradictory signals before cue generation

## Why it is designed this way
Full OpenXR session creation in Python depends on runtime + graphics bindings.
That part is real but still brittle across systems, so this version keeps the
OpenXR discovery layer real and the frame source swappable.

## Command examples
```bash
python aegisglyph_openxr_v2.py --mode auto --seconds 10 --hz 15
python aegisglyph_openxr_v2.py --mode probe --seconds 12 --hz 20 --log run.jsonl
python aegisglyph_openxr_v2.py --mode sim --seconds 20 --hz 12
```

## Real integration seam
Replace the synthetic frames in `ProbedOpenXRBackend.frames()` with real values
from a pyopenxr session based on the official `hello_xr` and pyopenxr example patterns.
