# Retinex

> Documentation group: Low-light enhancement (LLIE)

Retinex is a family of traditional low-light enhancement algorithms already implemented in openLLV. The current implementation includes SSR, MSR, and MSRCR.

## Links

| Type | URL |
| --- | --- |
| SSR-related paper | https://doi.org/10.1109/83.557356 |
| MSR / MSRCR-related paper | https://doi.org/10.1109/83.597272 |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

Retinex methods are based on the idea of separating illumination and reflectance. For low-light enhancement, Retinex suppresses slowly varying illumination components and highlights reflectance-related image details.

openLLV currently implements three Retinex variants:

| Algorithm | Meaning |
| --- | --- |
| `SSR` | Single Scale Retinex |
| `MSR` | Multi Scale Retinex |
| `MSRCR` | Multi Scale Retinex with Color Restoration |

## SSR

SSR uses one Gaussian surround scale to estimate illumination:

```text
R(x, y) = log(I(x, y)) - log(G_sigma(x, y) * I(x, y))
```

SSR is simple and efficient, but a single scale may not adapt well to both local details and global illumination.

## MSR

MSR averages Retinex responses over multiple Gaussian surround scales:

```text
MSR = mean(SSR_sigma_1, SSR_sigma_2, ..., SSR_sigma_n)
```

Compared with SSR, MSR balances detail enhancement and global tonal consistency more effectively.

## MSRCR

MSRCR adds a color restoration term on top of MSR to reduce color distortion caused by channel-wise Retinex processing. It is commonly used for color low-light images.

## Location in openLLV

| Item | Location |
| --- | --- |
| Algorithm implementation | `openLLV/tradition/algorithms/LLIE/Retinex.py` |
| SSR class name | `SSR` |
| MSR class name | `MSR` |
| MSRCR class name | `MSRCR` |
| Registered names | `SSR`, `MSR`, `MSRCR` |
| Aliases | `ssr`, `single_scale_retinex`, `msr`, `multi_scale_retinex`, `msrcr`, `multi_scale_retinex_color_restoration` |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Main Parameters

Common parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `low_clip` | `float` | `1.0` | Lower percentile used for display normalization |
| `high_clip` | `float` | `99.0` | Upper percentile used for display normalization |
| `eps` | `float` | `1e-6` | Small value used to avoid log and division instability |

SSR parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `sigma` | `float` | `80.0` | Gaussian surround scale |

MSR parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | Gaussian surround scales |

MSRCR parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | Gaussian surround scales |
| `alpha` | `float` | `125.0` | Color-restoration intensity gain |
| `beta` | `float` | `46.0` | Color-restoration log gain |
| `gain` | `float` | `1.0` | Global gain applied to the restored Retinex response |
| `offset` | `float` | `0.0` | Global offset applied before display normalization |

## Usage Examples

SSR:

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ssr",
    "input.jpg",
    output="results/ssr/output.jpg",
    sigma=80.0,
)
```

MSR:

```python
enhanced, saved_path = llv.predict(
    "msr",
    "input.jpg",
    output="results/msr/output.jpg",
    scales=(15.0, 80.0, 250.0),
)
```

MSRCR:

```python
enhanced, saved_path = llv.predict(
    "msrcr",
    "input.jpg",
    output="results/msrcr/output.jpg",
    alpha=125.0,
    beta=46.0,
)
```

Folder batch processing:

```python
saved_paths = llv.predict(
    "msrcr",
    "images/",
    output="results/msrcr",
    progress_bar=True,
)
```
