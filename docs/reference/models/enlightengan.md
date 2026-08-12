# EnlightenGAN

> Task: low-light image enhancement (LLIE)

EnlightenGAN is openLLV's attention-guided generator with global and local discriminators.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1109/TIP.2021.3051462 |
| Official source code | https://github.com/VITA-Group/EnlightenGAN |
| Official project page | https://github.com/VITA-Group/EnlightenGAN |
| Default configuration | `openLLV/deepLearning/config/EnlightenGAN.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/EnlightenGAN.py` |
| Class name | `EnlightenGAN` |
| Registered name | `EnlightenGAN` (the declared alias is identical; lookup is case-insensitive) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/EnlightenGAN_Loss.py` |

## Implementation Notes

The attention map is `(1 - channel_mean(input)).clamp(0, 1)`. In inference mode only the enhanced tensor is returned. Training mode returns the standardized dictionary and exposes the enhanced image, attention map, both discriminator modules, and a deterministic centered local crop. The crop side is at least 16 pixels where the input permits. The default YAML agrees with model defaults except that it omits `mode`, `model_name`, and `save_dir`.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration mapping; a non-dictionary raises `TypeError`; `**kwargs` override it. |
| `model_name` | `str` | `"EnlightenGAN"` | Base-class metadata. |
| `input_channels` | `int` | `3` | Input/output channels; must be a positive integer. |
| `save_dir` | `str` | `"./checkpoints/llie/EnlightenGAN"` | Default checkpoint directory. |
| `generator_channels` | `int` | `32` | Generator base width; converted to `int` and must be positive. |
| `discriminator_channels` | `int` | `32` | Base width for both discriminators; converted to `int` and must be positive. |
| `discriminator_layers` | `int` | `3` | Layer count for both discriminators; converted to `int` and must be positive. |
| `use_attention` | `bool` | `True` | Enables generator attention after boolean conversion. |
| `local_patch_ratio` | `float` | `0.5` | Center-crop ratio for local adversarial training; must be in `(0, 1]`. |
| `mode` | `str` | `"inference"` | Must be `"train"` or `"inference"`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "EnlightenGAN", "input.jpg", output="results/enlightengan/output.png",
    config={"use_attention": True, "local_patch_ratio": 0.4},
)
```

## Checkpoint / Official Weights

Use an openLLV checkpoint path as the prediction target. No automatic official-weight download is implemented.
