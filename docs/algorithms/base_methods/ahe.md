# AHE

> Documentation group: Base methods

AHE is an adaptive histogram equalization algorithm already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper / classic source | https://doi.org/10.1016/S0734-189X(87)80186-X |
| Paper page | https://www.sciencedirect.com/science/article/abs/pii/S0734189X8780186X |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

AHE, or Adaptive Histogram Equalization, is the local version of HE. Instead of performing one global histogram equalization over the whole image, it computes histogram mappings within local regions, so it can enhance local details and local contrast.

The main problem of AHE is that it can easily amplify noise in flat regions. CLAHE later mitigates this problem by limiting the local histogram clip limit.

## Location in openLLV

| Item | Location |
| --- | --- |
| Algorithm implementation | `openLLV/tradition/algorithms/BaseMethods/AHE.py` |
| Algorithm class name | `AHE` |
| Registered name | `ahe` |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

openLLV's AHE uses OpenCV's CLAHE interface and sets a large `clipLimit` to approximate AHE behavior.

Main parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | Color space where equalization is performed |
| `tile_grid_size` | `tuple` | `(8, 8)` | Local grid size |

Supported color spaces:

- `rgb` / `bgr`
- `hsv`
- `hls`
- `yuv` / `ycbcr`
- `lab`

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ahe",
    "input.jpg",
    output="results/ahe/output.jpg",
    color_space="yuv",
    tile_grid_size=(8, 8),
)
```
