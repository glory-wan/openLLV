# NPE

> Documentation group: traditional low-light image enhancement

NPE estimates bright-pass illumination, applies a bi-log mapping, restores reflectance detail, and blends toward the original image to preserve naturalness.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1109/TIP.2013.2261309 |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/LLIE/NPE.py` |
| Class name | `NPE` |
| Registered name | `NPE` (no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

Public three-channel input and NumPy output use RGB; NPE supports BGR/BGRA only as internal `_enhance()` layouts and also supports grayscale internally. It preserves grayscale/alpha layout. The base maps the resolved source range to NPE's `[0,1]` working range and restores the original value-range convention and optional dtype. All algorithm parameters are accepted as validated one-call `enhance()` overrides without mutating stored state. Factory creation ignores unsupported constructor keys after printing that they are unused and cannot affect computation; close spellings receive a suggestion.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `sigma` | `float` | `15.0` | Gaussian scale of bright-pass illumination smoothing. | Must be `> 0`. Runtime-overridable. |
| `illumination_floor` | `float` | `0.05` | Minimum illumination used in reflectance division. | Must be in `(0, 1]`. Runtime-overridable. |
| `enhancement_strength` | `float` | `4.0` | Bi-log illumination-mapping strength. | Must be `> 0`. Runtime-overridable. |
| `naturalness` | `float` | `0.35` | Original-image blend weight. | Must be in `[0, 1]`. Runtime-overridable. |
| `detail_weight` | `float` | `1.0` | Strength of the enhanced-minus-original detail term. | Must be `>= 0`. Runtime-overridable. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format. | Other values raise `ValueError`. |
| `keep_dtype` | `bool` | `True` | Cast output back to input dtype. | Non-`bool` raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip output to the resolved input value range. | Non-`bool` raises `TypeError`. |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | Input value range; auto distinguishes ordinary float `[0,1]` and `[0,255]` inputs. | Use `"byte"` when a byte-range float image has maximum `<= 1`. Custom bounds must be finite and increasing; out-of-range values raise `ValueError`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("NPE", "input.jpg", output="results/npe/output.png", enhancement_strength=5.0, naturalness=0.25)
```

```python
from openLLV.tradition.algorithms.LLIE.NPE import NPE

enhancer = NPE(sigma=12.0)
result = enhancer("input.jpg", detail_weight=1.2)
```
