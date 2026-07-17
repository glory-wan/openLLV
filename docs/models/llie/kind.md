# KinD

> Task: low-light image enhancement (`llie`)

KinD is a Retinex-inspired deep-learning model for practical low-light image enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1145/3343031.3350926 |
| Paper title | Kindling the Darkness: A Practical Low-light Image Enhancer |
| Official source code | https://github.com/zhangyhuaee/KinD |
| Official project page | None |

## Model Introduction

KinD decomposes an image into reflectance and illumination, restores degraded reflectance, adjusts illumination, and then recombines both components to produce the enhanced result. The original implementation trains the decomposition, restoration, and illumination adjustment networks in separate stages.

In openLLV, KinD is implemented as one integrated PyTorch model to match the unified training and prediction pipeline. The model contains:

| Subnetwork | Purpose |
| --- | --- |
| `KinDDecompositionNet` | Estimate reflectance `R` and illumination `I` |
| `KinDRestorationNet` | Restore low-light reflectance guided by illumination |
| `KinDIlluminationAdjustmentNet` | Adjust illumination using an exposure-ratio map |

The related loss combines the main objectives from the official staged training process, including Retinex reconstruction, reflectance consistency, illumination smoothness, reflectance restoration, illumination adjustment, and final enhanced-image supervision.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/KinD.py` |
| Model class name | `KinD` |
| Default configuration | `openLLV/deepLearning/config/KinD.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/KinD_Loss.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `decomposition_channels` | `int` | `64` | Feature channels used by the decomposition network |
| `decomposition_layers` | `int` | `5` | Number of intermediate decomposition layers |
| `restoration_channels` | `int` | `32` | Base feature channels used by the restoration network |
| `adjustment_channels` | `int` | `32` | Feature channels used by the illumination adjustment network |
| `adjustment_layers` | `int` | `3` | Number of intermediate illumination adjustment layers |
| `illumination_ratio` | `float` | `5.0` | Exposure ratio used by inference illumination adjustment |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "KinD",
    "input.jpg",
    output="results/KinD/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    "openLLV/deepLearning/config/KinD.yaml",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=2,
)
```

Override inference exposure ratio:

```python
enhanced, saved_path = llv.predict(
    "KinD",
    "input.jpg",
    output="results/KinD/brighter.png",
    device="cuda",
    illumination_ratio=6.0,
)
```
