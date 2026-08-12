# MSR

> Documentation group: Retinex low-light enhancement

MSR averages log-domain Retinex responses over multiple Gaussian surround scales.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1109/83.597272 |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/LLIE/Retinex.py` |
| Class name | `MSR` |
| Registered name | `MSR` (no aliases; lookup is case-insensitive) |
| Base class | `_RetinexBase` then `LLVEnhancer` |

## Implementation Notes

Each scale produces an SSR response; responses are averaged and percentile-normalized per channel. Grayscale and alpha layouts are preserved. `scales` may be overridden per enhancement call without mutating the configured tuple.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | Non-string, non-empty iterable of positive numeric-convertible values; stored as a float tuple. Invalid iterable/type raises `TypeError`; empty/non-positive raises `ValueError`. Per-call override supported. |
| `low_clip` | `float` | `1.0` | Lower percentile; require `0 <= low_clip < high_clip <= 100`. |
| `high_clip` | `float` | `99.0` | Upper percentile; same constraint. |
| `eps` | `float` | `1e-6` | Stabilizer; must be greater than zero. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format; invalid value raises `ValueError`. |
| `keep_dtype` | `bool` | `True` | Preserve input dtype; non-boolean raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip to valid dtype range; non-boolean raises `TypeError`. |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("MSR", "input.jpg", output="results/msr.png", scales=(10.0, 60.0, 180.0))
```
