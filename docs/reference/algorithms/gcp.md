# GCP

> Documentation group: traditional low-light image enhancement

GCP implements Gamma Correction Prior enhancement using adaptive gamma, atmospheric-light estimation, transmission recovery, and percentile range adjustment.

## Links

| Type | URL |
| --- | --- |
| Paper | https://www.sciencedirect.com/science/article/abs/pii/S0031320323006994 |
| Official source code | https://github.com/TripleJ2543/Low_Light_Pattern_Recognition_2023 |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/LLIE/GCP.py` |
| Class name | `GCP` |
| Registered name | `GCP` (aliases: `gcp`, `gcp-ms`; lookup is case-insensitive and trims whitespace, so `GCP` and `gcp` resolve to the same key) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

Public three-channel input and NumPy output use RGB; GCP uses BGR/BGRA only inside `_enhance()`. It converts grayscale and BGRA working input to three channels, preserves grayscale/alpha layout, and restores the source range. Each call starts with stored parameters, applies every runtime keyword as an override, then validates the combined mapping. Therefore unknown runtime keys are retained but not read after validation; known overrides do not mutate the instance. Factory construction ignores unsupported keys after printing that they are unused and cannot affect computation; close spellings receive a suggestion.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `gamma_max` | `float` | `6.0` | Maximum adaptive gamma. | Must be `>= 1`. Runtime-overridable. |
| `erosion_window` | `int` | `15` | Dark-channel erosion kernel size. | Must be `> 0`. Runtime-overridable. |
| `atmospheric_bins` | `int` | `200` | Histogram bins for atmospheric-light estimation. | Must be `> 0`. Runtime-overridable. |
| `atmospheric_percentile` | `float` | `0.99` | Dark-channel fraction used to select candidates. | Must be in `(0, 1)`. Runtime-overridable. |
| `t_min` | `float` | `0.1` | Transmission floor. | Must be in `(0, 1]`. Runtime-overridable. |
| `blur_ksize` | `int` | `7` | Gaussian blur kernel size. | Must be a positive odd integer. Runtime-overridable. |
| `high_percentile` | `float` | `99.5` | Upper percentile for final range adjustment. | With `low_percentile`, must satisfy `0 <= low < high <= 100`. Runtime-overridable. |
| `low_percentile` | `float` | `0.5` | Lower percentile for final range adjustment. | With `high_percentile`, must satisfy `0 <= low < high <= 100`. Runtime-overridable. |
| `eps` | `float` | `0.000001` | Numerical-stability floor. | Must be `> 0`. Runtime-overridable. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format. | Other values raise `ValueError`. |
| `keep_dtype` | `bool` | `True` | Cast output back to input dtype. | Non-`bool` raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip output to the resolved input value range. | Non-`bool` raises `TypeError`. |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | Input value range; auto distinguishes ordinary float `[0,1]` and `[0,255]` inputs. | Use `"byte"` when a byte-range float image has maximum `<= 1`. Custom bounds must be finite and increasing; out-of-range values raise `ValueError`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("gcp-ms", "input.jpg", output="results/gcp/output.png", gamma_max=5.0, blur_ksize=5)
```

```python
from openLLV.tradition.algorithms.LLIE.GCP import GCP

enhancer = GCP(high_percentile=99.0)
result = enhancer("input.jpg", low_percentile=1.0)
```
