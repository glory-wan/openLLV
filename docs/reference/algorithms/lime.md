# LIME

> Documentation group: traditional low-light image enhancement

LIME estimates illumination from the per-pixel channel maximum, refines it with a guided filter, and divides the image by the gamma-adjusted illumination.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1109/TIP.2016.2639450 |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/LLIE/LIME.py` |
| Class name | `LIME` |
| Registered name | `LIME` (no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

Public three-channel input and NumPy output use RGB; LIME supports BGR/BGRA only as internal `_enhance()` layouts and also supports grayscale internally. It preserves grayscale/alpha layout. The base maps the resolved source range to LIME's `[0,1]` working range, then restores the original value-range convention and optional dtype. Each algorithm parameter may be supplied to `enhance()` for one call, where it is validated without mutating stored state. Factory creation ignores unsupported constructor keys after printing that they are unused and cannot affect computation; close spellings receive a suggestion.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `gamma` | `float` | `0.8` | Exponent applied to refined illumination. | Must be `> 0`. Runtime-overridable. |
| `guided_radius` | `int` | `15` | Guided-filter box size. | Must be `> 0`. Runtime-overridable. |
| `guided_eps` | `float` | `0.001` | Guided-filter regularization. | Must be `> 0`. Runtime-overridable. |
| `illumination_floor` | `float` | `0.05` | Lower bound applied before division. | Must be in `(0, 1]`. Runtime-overridable. |
| `exposure` | `float` | `1.0` | Global multiplier after illumination correction. | Must be `> 0`. Runtime-overridable. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format. | Other values raise `ValueError`. |
| `keep_dtype` | `bool` | `True` | Cast output back to input dtype. | Non-`bool` raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip output to the resolved input value range. | Non-`bool` raises `TypeError`. |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | Input value range; auto distinguishes ordinary float `[0,1]` and `[0,255]` inputs. | Use `"byte"` when a byte-range float image has maximum `<= 1`. Custom bounds must be finite and increasing; out-of-range values raise `ValueError`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("LIME", "input.jpg", output="results/lime/output.png", gamma=0.9, exposure=1.1)
```

```python
from openLLV.tradition.algorithms.LLIE.LIME import LIME

enhancer = LIME(guided_radius=21)
result = enhancer("input.jpg", illumination_floor=0.08)
```
