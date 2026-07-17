# GCP

> Documentation group: Low-light enhancement (LLIE)

GCP is a Gamma Correction Prior low-light enhancement algorithm implemented in openLLV, corresponding to the paper “Low-light image enhancement using gamma correction prior in mixed color spaces”.

## Links

| Type | URL |
| --- | --- |
| Paper link | https://www.sciencedirect.com/science/article/abs/pii/S0031320323006994 |
| Official source code | https://github.com/TripleJ2543/Low_Light_Pattern_Recognition_2023 |
| Official project page | None |

## Algorithm Introduction

GCP is not ordinary Gamma correction with a fixed exponent. It uses a gamma correction prior to construct pixel-level adaptive gamma and combines the dark channel idea to estimate atmospheric light and the transmission map, thereby improving brightness and visible details in low-light images.

The openLLV implementation refers to the official open-source script. The main pipeline includes:

1. Convert the image to normalized floating-point space.
2. Apply Gaussian smoothing to the input image and invert it to obtain a low-light degradation representation for estimation.
3. Compute the dark channel and estimate atmospheric light from regions with high dark-channel responses.
4. Normalize channels according to atmospheric light.
5. Generate adaptive gamma from the maximum channel value of each pixel and estimate the transmission map.
6. Recover the enhanced image from the transmission map and use percentile dynamic-range stretching to obtain the final result.

## Location in openLLV

| Item | Location |
| --- | --- |
| Algorithm implementation | `openLLV/tradition/algorithms/LLIE/GCP.py` |
| Algorithm class name | `GCP` |
| Registered name | `GCP` |
| Aliases | `gcp`, `gamma_correction_prior` |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `gamma_max` | `float` | `6.0` | Maximum value of pixel-adaptive gamma |
| `erosion_window` | `int` | `15` | Dark-channel erosion kernel size |
| `atmospheric_bins` | `int` | `200` | Number of histogram bins used during atmospheric light estimation |
| `atmospheric_percentile` | `float` | `0.99` | Dark-channel percentile ratio used to select atmospheric-light candidate regions |
| `t_min` | `float` | `0.1` | Lower bound of the transmission map |
| `blur_ksize` | `int` | `7` | Gaussian smoothing kernel size, must be a positive odd integer |
| `high_percentile` | `float` | `99.5` | High percentile for final dynamic-range stretching |
| `low_percentile` | `float` | `0.5` | Low percentile for final dynamic-range stretching |
| `eps` | `float` | `1e-6` | Small value to avoid division by zero |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "gcp",
    "input.jpg",
    output="results/gcp/output.jpg",
)
```

Explicitly call the traditional algorithm backend through the unified Predictor:

```python
from openLLV import Predictor

predictor = Predictor("gcp", backend="traditional")
predictor("images/", output="results/gcp")
```

Pass custom parameters:

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "gcp",
    "input.jpg",
    output="results/gcp_custom.png",
    gamma_max=5.0,
    erosion_window=11,
    high_percentile=99.0,
    low_percentile=1.0,
)
```
