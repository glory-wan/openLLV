# RCLAHE

> Documentation group: Base methods

RCLAHE is a recursive CLAHE enhancement algorithm already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Base algorithm paper / classic source | https://ieeexplore.ieee.org/document/109340/ |
| CLAHE Graphics Gems chapter | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

RCLAHE, or Recursive CLAHE, is a recursive enhancement version based on CLAHE. It repeatedly applies CLAHE to the input image and gradually enhances local contrast multiple times.

Compared with single-pass CLAHE, RCLAHE can achieve stronger local enhancement, but too many iterations may cause over-enhancement, noise amplification, or unnatural colors. Therefore, `iterations` and `clip_limit` should be controlled in practical use.

## Location in openLLV

| Item | Location |
| --- | --- |
| Algorithm implementation | `openLLV/tradition/algorithms/BaseMethods/RCLAHE.py` |
| Algorithm class name | `RCLAHE` |
| Registered name | `rclahe` |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

openLLV's RCLAHE repeatedly calls the CLAHE processing pipeline inside `_enhance()`.

Main parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | Color space where CLAHE is performed |
| `clip_limit` | `float` | `2.0` | Contrast clipping threshold |
| `tile_grid_size` | `tuple` | `(8, 8)` | Local grid size |
| `iterations` | `int` | `2` | Number of recursive CLAHE applications |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "rclahe",
    "input.jpg",
    output="results/rclahe/output.jpg",
    color_space="hsv",
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    iterations=3,
)
```
