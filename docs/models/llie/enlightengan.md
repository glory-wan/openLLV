# EnlightenGAN

> Task: low-light image enhancement (`llie`)

EnlightenGAN is an attention-guided generative adversarial model for unpaired low-light image enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1109/TIP.2021.3051462 |
| Paper title | EnlightenGAN: Deep Light Enhancement Without Paired Supervision |
| Official source code | https://github.com/VITA-Group/EnlightenGAN |
| Official project page | https://github.com/VITA-Group/EnlightenGAN |

## Model Introduction

EnlightenGAN enhances low-light images without requiring paired low-light and normal-light supervision. The original method uses an attention-guided U-Net generator, global and local discriminators, and self feature preserving regularization so that enhanced images become brighter while preserving the structure and color characteristics of the input.

In openLLV, EnlightenGAN is adapted to the unified model and trainer interface. The model contains:

| Component | Purpose |
| --- | --- |
| `EnlightenGANGenerator` | Attention-guided U-Net generator |
| `global_discriminator` | PatchGAN discriminator for full images |
| `local_discriminator` | PatchGAN discriminator for cropped local patches |
| attention map | Darkness-aware map estimated from the input image |

The related loss combines global/local LSGAN losses, discriminator losses, self feature-preserving regularization, exposure regularization, and total variation smoothness.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/EnlightenGAN.py` |
| Model class name | `EnlightenGAN` |
| Default configuration | `openLLV/deepLearning/config/EnlightenGAN.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/EnlightenGAN_Loss.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `generator_channels` | `int` | `32` | Base feature channels used by the generator |
| `discriminator_channels` | `int` | `32` | Base feature channels used by discriminators |
| `discriminator_layers` | `int` | `3` | Number of PatchGAN downsampling layers |
| `use_attention` | `bool` | `True` | Whether to concatenate the attention map to the generator input |
| `local_patch_ratio` | `float` | `0.5` | Ratio of the center local patch used by local adversarial loss |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "EnlightenGAN",
    "input.jpg",
    output="results/EnlightenGAN/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    "openLLV/deepLearning/config/EnlightenGAN.yaml",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=2,
)
```

Quick debug configuration:

```python
llv.train(
    model="EnlightenGAN",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="enlightengan",
    model_params={
        "generator_channels": 16,
        "discriminator_channels": 16,
    },
    epochs=2,
    batch_size=1,
)
```
