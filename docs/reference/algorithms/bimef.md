# BIMEF

> Documentation group: traditional low-light image enhancement

BIMEF fuses the original image with an automatically or manually exposed copy using contrast, saturation, and well-exposedness weights.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.48550/arXiv.1711.00591 |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/LLIE/BIMEF.py` |
| Class name | `BIMEF` |
| Registered name | `BIMEF` (no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

The algorithm accepts BGR, grayscale, and BGRA arrays, normalizes them to `[0, 1]`, preserves alpha/channel layout, and restores the source value range. When `exposure_ratio=None`, it estimates `clip(target_mean / mean_luminance, 1, max_ratio)`. Every algorithm parameter can also be passed to `enhance()` as a one-call override; overrides are validated and do not mutate stored parameters. Factory creation filters constructor keys and warns about unsupported keys.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `exposure_ratio` | `Optional[float]` | `None` | Manual exposure multiplier; `None` enables estimation. | If not `None`, must be `> 0`. Runtime-overridable. |
| `target_mean` | `float` | `0.55` | Target mean luminance for automatic exposure. | Must be in `(0, 1)`. Runtime-overridable. |
| `max_ratio` | `float` | `5.0` | Upper bound for automatic exposure. | Must be `>= 1`. Runtime-overridable. |
| `well_exposed_sigma` | `float` | `0.2` | Gaussian sigma of the well-exposedness weight. | Must be `> 0`. Runtime-overridable. |
| `contrast_weight` | `float` | `1.0` | Contrast-weight exponent. | Must be `>= 0`. Runtime-overridable. |
| `saturation_weight` | `float` | `1.0` | Saturation-weight exponent. | Must be `>= 0`. Runtime-overridable. |
| `well_exposed_weight` | `float` | `1.0` | Well-exposedness exponent. | Must be `>= 0`. Runtime-overridable. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format. | Other values raise `ValueError`. |
| `keep_dtype` | `bool` | `True` | Cast output back to input dtype. | Non-`bool` raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip output to the destination dtype range. | Non-`bool` raises `TypeError`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("BIMEF", "input.jpg", output="results/bimef/output.png", exposure_ratio=2.0, target_mean=0.6)
```

```python
from openLLV.tradition.algorithms.LLIE.BIMEF import BIMEF

enhancer = BIMEF(max_ratio=4.0)
result = enhancer("input.jpg", contrast_weight=0.8)  # one-call override
```
