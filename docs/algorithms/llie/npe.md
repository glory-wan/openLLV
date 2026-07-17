# NPE

> Documentation group: Low-light enhancement (LLIE)

NPE is a Naturalness Preserved Enhancement algorithm for non-uniform illumination images.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1109/TIP.2013.2261309 |
| Paper title | Naturalness Preserved Enhancement Algorithm for Non-Uniform Illumination Images |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

NPE enhances images under non-uniform illumination while preserving a natural visual appearance. The method estimates illumination, separates reflectance-like details, maps illumination with a naturalness-preserving transform, and blends enhanced details with the original image.

In openLLV, NPE uses a bright-pass illumination estimate, Gaussian smoothing, a bi-log style illumination transform, and a naturalness blending weight.

## Location in openLLV

| Item | Location |
| --- | --- |
| Algorithm implementation | `openLLV/tradition/algorithms/LLIE/NPE.py` |
| Algorithm class name | `NPE` |
| Registered name | `NPE` |
| Aliases | `npe`, `naturalness_preserved_enhancement` |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `sigma` | `float` | `15.0` | Gaussian scale for bright-pass illumination filtering |
| `illumination_floor` | `float` | `0.05` | Lower bound for illumination |
| `enhancement_strength` | `float` | `4.0` | Strength of the bi-log illumination mapping |
| `naturalness` | `float` | `0.35` | Blend weight for preserving the original naturalness |
| `detail_weight` | `float` | `1.0` | Weight applied to reflectance detail restoration |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "npe",
    "input.jpg",
    output="results/npe/output.jpg",
)
```

Adjust naturalness:

```python
enhanced, saved_path = llv.predict(
    "npe",
    "input.jpg",
    output="results/npe/natural.jpg",
    naturalness=0.5,
    enhancement_strength=3.0,
)
```
