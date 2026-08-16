# HE

> Documentation group: base methods

HE applies global histogram equalization to grayscale, internally converted BGR channels, or a selected luminance channel; its public NumPy contract is RGB.

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

Public three-channel input and NumPy output use RGB. `LLVEnhancer` converts to BGR before `_enhance()` and back to RGB afterward. Input is converted to `uint8` for OpenCV. Grayscale is equalized directly; internally, `rgb` means per-channel BGR equalization, while `hsv`, `hls`, `yuv`, and `lab` process V, L, Y, and L. Per-call algorithm kwargs are ignored.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `color_space` | `str` | `"rgb"` | `rgb`/`bgr`, `hsv`, `hls`, `yuv`/`ycbcr`, or `lab`; case-insensitive. Invalid type/value raises `TypeError`/`ValueError`. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format; invalid value raises `ValueError`. |
| `keep_dtype` | `bool` | `True` | Preserve input dtype; non-boolean raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip to the resolved input value range; non-boolean raises `TypeError`. |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | Input value range. Auto infers float `[0,1]` or `[0,255]`; use `"byte"` for ambiguous dark byte-range floats whose maximum is `<= 1`. Custom bounds must be finite and increasing. Values outside the selected range raise `ValueError`. |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("he", "input.jpg", output="results/he.png", color_space="lab")
```
