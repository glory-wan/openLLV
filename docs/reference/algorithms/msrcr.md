# MSRCR

> Documentation group: Retinex low-light enhancement

MSRCR adds channel color restoration and global gain/offset to Multi Scale Retinex.

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
| Class name | `MSRCR` |
| Registered name | `MSRCR` (no aliases; lookup is case-insensitive) |
| Base class | `MSR` → `_RetinexBase` → `LLVEnhancer` |

## Implementation Notes

After MSR, color input receives `beta * (log(alpha*channel+eps)-log(channel_sum+eps))`; the restored response is `gain * (color_restoration * retinex + offset)`. Grayscale uses `gain * (retinex + offset)`. Output is percentile-normalized; alpha is preserved. All MSRCR-specific values and `scales` accept per-call overrides without instance mutation.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | Non-empty iterable of positive numeric-convertible values; per-call override supported. |
| `alpha` | `float` | `125.0` | Color-restoration intensity gain; numeric and greater than zero. Per-call override supported. |
| `beta` | `float` | `46.0` | Color-restoration log gain; numeric and greater than zero. Per-call override supported. |
| `gain` | `float` | `1.0` | Global gain; numeric and greater than zero. Per-call override supported. |
| `offset` | `float` | `0.0` | Global offset; numeric with no range constraint. Per-call override supported. |
| `low_clip` | `float` | `1.0` | Lower percentile; require `0 <= low_clip < high_clip <= 100`. |
| `high_clip` | `float` | `99.0` | Upper percentile; same constraint. |
| `eps` | `float` | `1e-6` | Stabilizer; must be greater than zero. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format; invalid value raises `ValueError`. |
| `keep_dtype` | `bool` | `True` | Preserve input dtype; non-boolean raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip to valid dtype range; non-boolean raises `TypeError`. |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("MSRCR", "input.jpg", output="results/msrcr.png", scales=(15.0, 100.0), alpha=100.0, beta=40.0)
```
