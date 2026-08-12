# HE

> Documentation group: base methods

HE applies global histogram equalization to grayscale, individual BGR channels, or a selected luminance channel.

## Links

| Type | URL |
| --- | --- |
| Paper | None |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/BaseMethods/HE.py` |
| Class name | `HE` |
| Registered name | `he` (class name also registers; no aliases; lookup is case-insensitive) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

Input is converted to `uint8` for OpenCV. Grayscale is equalized directly; `rgb` means per-channel BGR equalization, while `hsv`, `hls`, `yuv`, and `lab` process V, L, Y, and L. Per-call algorithm kwargs are ignored.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `color_space` | `str` | `"rgb"` | `rgb`/`bgr`, `hsv`, `hls`, `yuv`/`ycbcr`, or `lab`; case-insensitive. Invalid type/value raises `TypeError`/`ValueError`. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format; invalid value raises `ValueError`. |
| `keep_dtype` | `bool` | `True` | Preserve input dtype; non-boolean raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip to valid dtype range; non-boolean raises `TypeError`. |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("he", "input.jpg", output="results/he.png", color_space="lab")
```
