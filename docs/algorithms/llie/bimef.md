# BIMEF

> Documentation group: Low-light enhancement (LLIE)

BIMEF is a bio-inspired multi-exposure fusion framework for low-light image enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.48550/arXiv.1711.00591 |
| Paper title | A Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

BIMEF enhances low-light images by creating an exposure-adjusted version of the input image and fusing it with the original image. Fusion weights are computed from contrast, saturation, and well-exposedness so that details and natural appearance can be balanced.

In openLLV, the exposure ratio can be manually specified or automatically estimated from the image luminance mean.

## Location in openLLV

| Item | Location |
| --- | --- |
| Algorithm implementation | `openLLV/tradition/algorithms/LLIE/BIMEF.py` |
| Algorithm class name | `BIMEF` |
| Registered name | `BIMEF` |
| Aliases | `bimef`, `bio_inspired_multi_exposure_fusion` |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `exposure_ratio` | `Optional[float]` | `None` | Manual exposure ratio; if `None`, it is estimated automatically |
| `target_mean` | `float` | `0.55` | Target luminance mean for automatic exposure |
| `max_ratio` | `float` | `5.0` | Maximum automatic exposure ratio |
| `well_exposed_sigma` | `float` | `0.2` | Sigma for well-exposedness weight |
| `contrast_weight` | `float` | `1.0` | Contrast weight exponent |
| `saturation_weight` | `float` | `1.0` | Saturation weight exponent |
| `well_exposed_weight` | `float` | `1.0` | Well-exposedness weight exponent |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "bimef",
    "input.jpg",
    output="results/bimef/output.jpg",
)
```

Manual exposure:

```python
enhanced, saved_path = llv.predict(
    "bimef",
    "input.jpg",
    output="results/bimef/manual.jpg",
    exposure_ratio=3.0,
)
```
