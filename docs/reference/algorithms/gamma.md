# Gamma

> Documentation group: traditional low-light image enhancement

Gamma performs channel-wise power-law correction after normalizing the image to `[0, 1]`.

## Links

| Type | URL |
| --- | --- |
| Paper | None |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/LLIE/Gamma.py` |
| Class name | `Gamma` |
| Registered name | `Gamma` (no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

Public three-channel input and NumPy output use RGB; the base class performs the internal BGR round trip around `_enhance()`. The base resolves the semantic input range, maps it to Gamma's `[0,1]` floating working range, and restores the same source range and dtype afterward. Thus float `[0,1]` and float `[0,255]` inputs retain their respective output conventions. A `gamma` passed to `enhance()` overrides the stored value for that call without mutation. Values below `1` brighten the image. Factory creation ignores unsupported constructor keys after printing that they are unused and cannot affect computation; close spellings receive a suggestion.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `gamma` | `float` | `0.6` | Power-law exponent. | Must be an `int` or `float` and `> 0`; otherwise `TypeError` or `ValueError`. Runtime-overridable. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format. | Other values raise `ValueError`. |
| `keep_dtype` | `bool` | `True` | Cast output back to input dtype. | Non-`bool` raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip output to the resolved input value range. | Non-`bool` raises `TypeError`. |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | Input value range; auto distinguishes ordinary float `[0,1]` and `[0,255]` inputs. | Use `"byte"` when a byte-range float image has maximum `<= 1`. Custom bounds must be finite and increasing; out-of-range values raise `ValueError`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("Gamma", "input.jpg", output="results/gamma/output.png", gamma=0.45)
```

```python
from openLLV.tradition.algorithms.LLIE.Gamma import Gamma

enhancer = Gamma(gamma=0.7)
result = enhancer("input.jpg", gamma=0.5)
```
