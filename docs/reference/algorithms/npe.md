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

NPE supports BGR, grayscale, and BGRA arrays and preserves grayscale/alpha layout. It normalizes integer input, clips float input to `[0, 1]`, and restores the source range. All algorithm parameters are accepted as validated one-call `enhance()` overrides without mutating stored state. Factory creation filters unsupported constructor keys and warns.

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
| `clip_output` | `bool` | `True` | Clip output to the destination dtype range. | Non-`bool` raises `TypeError`. |

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
