# CLAHE

> Documentation group: Base methods

CLAHE is a contrast-limited adaptive histogram equalization algorithm already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper / classic source | https://ieeexplore.ieee.org/document/109340/ |
| Graphics Gems chapter | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

CLAHE, or Contrast Limited Adaptive Histogram Equalization, is an improved version of AHE. It introduces a clip limit on top of local histogram equalization to constrain local contrast amplification, reducing the problem of over-enhanced noise.

CLAHE is commonly used in medical images, low-light images, remote-sensing images, and other low-contrast image enhancement tasks. Compared with HE, CLAHE focuses more on local details; compared with AHE, CLAHE is more robust to noise.

## Location in openLLV

| Item | Location |
| --- | --- |
| Algorithm implementation | `openLLV/tradition/algorithms/BaseMethods/CLAHE.py` |
| Algorithm class name | `CLAHE` |
| Registered name | `clahe` |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

openLLV's CLAHE is based on OpenCV's `cv2.createCLAHE()` and supports processing luminance channels or all channels in different color spaces.

Main parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | Color space where CLAHE is performed |
| `clip_limit` | `float` | `2.0` | Contrast clipping threshold |
| `tile_grid_size` | `tuple` | `(8, 8)` | Local grid size |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "clahe",
    "input.jpg",
    output="results/clahe/output.jpg",
    color_space="lab",
    clip_limit=2.0,
    tile_grid_size=(8, 8),
)
```

Folder batch processing:

```python
saved_paths = llv.predict(
    "clahe",
    "images/",
    output="results/clahe",
    progress_bar=True,
)
```
