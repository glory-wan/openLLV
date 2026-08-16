# SSR

> Documentation group: Retinex low-light enhancement

SSR computes a log-domain Single Scale Retinex response and percentile-normalizes it for display.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1109/83.557356 |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/LLIE/Retinex.py` |
| Class name | `SSR` |
| Registered name | `SSR` (no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `_RetinexBase` then `LLVEnhancer`, both in `openLLV/tradition/algorithms/LLIE/Retinex.py` / `BaseModel.py` |

## Implementation Notes

Public three-channel input and NumPy output use RGB; the base class performs the internal BGR round trip around `_enhance()`. The base maps the resolved source range to SSR's `[0,1]` working range and restores the original value-range convention and optional dtype. The response is `log(image+eps)-log(GaussianBlur(image)+eps)`, normalized per channel between configured percentiles. Grayscale layout is preserved, as is a fourth alpha channel. `sigma` may be overridden per enhancement call without mutating the instance.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `sigma` | `float` | `80.0` | Gaussian surround scale; must be numeric and positive (`bool` passes Python's numeric check). Per-call override supported. |
| `low_clip` | `float` | `1.0` | Lower normalization percentile. Together with `high_clip`, must satisfy `0 <= low_clip < high_clip <= 100`. |
| `high_clip` | `float` | `99.0` | Upper normalization percentile; same constraint. |
| `eps` | `float` | `1e-6` | Log/division stabilizer; must be greater than zero. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format; invalid value raises `ValueError`. |
| `keep_dtype` | `bool` | `True` | Preserve input dtype; non-boolean raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip to the resolved input value range; non-boolean raises `TypeError`. |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | Input value range. Auto infers float `[0,1]` or `[0,255]`; use `"byte"` for ambiguous dark byte-range floats whose maximum is `<= 1`. Custom bounds must be finite and increasing. Values outside the selected range raise `ValueError`. |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("SSR", "input.jpg", output="results/ssr.png", sigma=100.0, low_clip=0.5, high_clip=99.5)
```
