# RCLAHE

> Documentation group: base methods

RCLAHE recursively applies the same CLAHE operation to strengthen local contrast.

## Links

| Type | URL |
| --- | --- |
| Paper | None |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/BaseMethods/RCLAHE.py` |
| Class name | `RCLAHE` |
| Registered name | `rclahe` (class name also registers; no aliases; lookup is case-insensitive) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

Input is converted to `uint8`, then CLAHE is repeated `iterations` times. Channel behavior matches CLAHE. Per-call algorithm kwargs are ignored. `set_params` does not rebuild the cached OpenCV object, so reconstruct the enhancer after changing `clip_limit` or `tile_grid_size`; changing `iterations` does take effect.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | `rgb`/`bgr`, `hsv`, `hls`, `yuv`/`ycbcr`, or `lab`; invalid type/value raises `TypeError`/`ValueError`. |
| `clip_limit` | `float` | `2.0` | Finite numeric, non-boolean, greater than zero; otherwise `ValueError`. |
| `tile_grid_size` | `Tuple[int, int]` | `(8, 8)` | Exactly two positive non-boolean integers; otherwise `ValueError`. |
| `iterations` | `int` | `3` | Positive non-boolean integer; otherwise `ValueError`. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format; invalid value raises `ValueError`. |
| `keep_dtype` | `bool` | `True` | Preserve input dtype; non-boolean raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip to valid dtype range; non-boolean raises `TypeError`. |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("rclahe", "input.jpg", output="results/rclahe.png", clip_limit=2.5, iterations=2)
```
