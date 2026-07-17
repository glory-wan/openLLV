# KinD++

> Task: low-light image enhancement (`llie`)

KinD++ is an improved Retinex-based low-light enhancement model that extends KinD with multi-scale illumination attention.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1007/s11263-020-01407-x |
| Paper title | Beyond Brightening Low-light Images |
| Official source code | https://github.com/zhangyhuaee/KinD_plus |
| Official project page | None |

## Model Introduction

KinD++ improves the original KinD framework for more robust low-light enhancement. It keeps the Retinex-style decomposition-restoration-adjustment pipeline, but introduces a multi-scale illumination attention module (MSIA) in the reflectance restoration network. MSIA uses the estimated illumination map to guide multi-scale feature restoration, reducing non-uniform spots and over-smoothing artifacts.

In openLLV, the model is implemented as `KinDPlusPlus` because Python class names cannot contain `++`. The default YAML file keeps the paper-style name:

```text
openLLV/deepLearning/config/KinD++.yaml
```

The openLLV implementation contains:

| Subnetwork | Purpose |
| --- | --- |
| `KinDPPDecompositionNet` | Estimate reflectance and illumination |
| `KinDPPRestorationNet` | Restore reflectance with MSIA-guided features |
| `KinDPPIlluminationAdjustmentNet` | Adjust illumination with an exposure-ratio map |

The related loss combines the main objectives used by the official staged training code: decomposition reconstruction, reflectance consistency, mutual illumination constraints, input-aware illumination smoothness, reflectance restoration, illumination adjustment, and final enhanced-image supervision.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/KinDPlusPlus.py` |
| Model class name | `KinDPlusPlus` |
| Default configuration | `openLLV/deepLearning/config/KinD++.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/KinDPlusPlus_Loss.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `decomposition_channels` | `int` | `32` | Base feature channels used by the decomposition network |
| `restoration_channels` | `int` | `32` | Base feature channels used by the MSIA restoration network |
| `adjustment_channels` | `int` | `32` | Feature channels used by the illumination adjustment network |
| `illumination_ratio` | `float` | `5.0` | Exposure ratio used by inference illumination adjustment |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "KinDPlusPlus",
    "input.jpg",
    output="results/KinDPlusPlus/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    "openLLV/deepLearning/config/KinD++.yaml",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=2,
)
```

Override inference exposure ratio:

```python
enhanced, saved_path = llv.predict(
    "KinDPlusPlus",
    "input.jpg",
    output="results/KinDPlusPlus/brighter.png",
    device="cuda",
    illumination_ratio=6.0,
)
```
