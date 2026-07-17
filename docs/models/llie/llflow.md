# LLFlow

> Task: low-light image enhancement (`llie`)

LLFlow is a conditional normalizing-flow model for low-light image enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1609/aaai.v36i3.20162 |
| Paper title | Low-Light Image Enhancement with Normalizing Flow |
| Official source code | https://github.com/wyf0912/LLFlow |
| Official project page | https://github.com/wyf0912/LLFlow |

## Model Introduction

LLFlow models the conditional distribution of normally exposed images given a low-light image with a normalizing flow. Instead of predicting a single deterministic output only, it learns an invertible mapping between image space and latent space conditioned on the low-light input. The original method trains the flow with a negative log-likelihood objective and can generate enhanced images by sampling or using a deterministic latent code.

In openLLV, LLFlow is adapted to the unified image-to-image model interface. The implementation contains:

| Component | Purpose |
| --- | --- |
| `LLFlowConditionEncoder` | Extract low-light conditional features |
| `LLFlowAffineCoupling` | Conditional affine coupling flow layer |
| `LLFlow` | Stack conditional flow layers and run forward/reverse transforms |

During training, the loss maps the paired normal-light target into latent space and computes a flow negative log-likelihood. During inference, the model uses a zero latent tensor by default to produce deterministic enhancement. Set `sample_temperature > 0` to sample stochastic outputs.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/LLFlow.py` |
| Model class name | `LLFlow` |
| Default configuration | `openLLV/deepLearning/config/LLFlow.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/LLFlow_Loss.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `condition_channels` | `int` | `32` | Number of low-light conditional feature channels |
| `condition_blocks` | `int` | `4` | Number of residual blocks in the condition encoder |
| `flow_layers` | `int` | `8` | Number of conditional affine coupling layers |
| `flow_hidden_channels` | `int` | `64` | Hidden channels in each coupling network |
| `scale_clamp` | `float` | `2.0` | Clamp range for affine coupling log-scale |
| `sample_temperature` | `float` | `0.0` | Latent sampling temperature used during inference |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LLFlow",
    "input.jpg",
    output="results/LLFlow/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    "openLLV/deepLearning/config/LLFlow.yaml",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=2,
)
```

Stochastic inference:

```python
enhanced, saved_path = llv.predict(
    "LLFlow",
    "input.jpg",
    output="results/LLFlow/sample.png",
    device="cuda",
    sample_temperature=0.7,
)
```
