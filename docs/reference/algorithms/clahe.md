# CLAHE

> Documentation group: base methods

CLAHE performs contrast-limited local histogram equalization with OpenCV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/BaseMethods/CLAHE.py` |
| Class name | `CLAHE` |
| Registered name | `clahe` (class name also registers; no aliases; lookup is case-insensitive) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

Grayscale input is equalized directly. For BGR input, `rgb` processes every channel and other modes process their luminance/value channel. Non-`uint8` input is converted first. Per-call algorithm kwargs are ignored. Because the OpenCV object is cached at construction, changing `clip_limit` or `tile_grid_size` through `set_params` does not rebuild it; create a new enhancer instead.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | `rgb`/`bgr`, `hsv`, `hls`, `yuv`/`ycbcr`, or `lab`; case-insensitive. Invalid type/value raises `TypeError`/`ValueError`. |
| `clip_limit` | `float` | `2.0` | Finite numeric, non-boolean, greater than zero; otherwise `ValueError`. |
| `tile_grid_size` | `Tuple[int, int]` | `(8, 8)` | Exactly two positive non-boolean integers; otherwise `ValueError`. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format; invalid value raises `ValueError`. |
| `keep_dtype` | `bool` | `True` | Preserve input dtype; non-boolean raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip to valid dtype range; non-boolean raises `TypeError`. |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("clahe", "input.jpg", output="results/clahe.png", color_space="hsv", clip_limit=3.0, tile_grid_size=(4, 4))
```
