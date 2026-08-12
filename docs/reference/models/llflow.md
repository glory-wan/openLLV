# LLFlow

> Task: low-light image enhancement (LLIE)

LLFlow is a conditional normalizing-flow model registered in openLLV for low-light image enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1609/aaai.v36i3.20162 |
| Official source code | https://github.com/wyf0912/LLFlow |
| Official project page | https://wyf0912.github.io/LLFlow/ |
| Default configuration | `openLLV/deepLearning/config/LLFlow.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/LLFlow.py` |
| Class name | `LLFlow` |
| Registered name | `LLFlow` (no aliases) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/LLFlow_Loss.py` (`llflow`; aliases: `llflow_loss`, `low_light_flow`, `normalizing_flow_loss`) |

## Implementation Notes

The condition encoder extracts low-light features and alternating affine-coupling layers map between image and latent spaces. The input must have at least two channels. In inference mode, the latent is all zeros when `sample_temperature=0.0`, otherwise Gaussian noise scaled by the temperature. In training mode, `forward()` returns `{"pred", "aux", "meta"}`; `aux` includes the condition, latent, and callable forward/reverse flow transforms needed by `LLFlow_Loss`. In inference mode it returns only the enhanced tensor in `[0, 1]`.

The constructor is `LLFlow(config=None, **kwargs)`. Defaults are merged first, then `config`, then `**kwargs`; keyword overrides therefore win. `config` must be a dictionary or `None`. Unknown keys are retained in `model.config` but are not consumed by this model.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Model configuration dictionary. | A non-dictionary, non-`None` value raises `TypeError`. |
| `model_name` | `str` | `"LLFlow"` | Stored model name inherited from `LLVModel`. | Not otherwise validated. |
| `input_channels` | `int` | `3` | Input and flow channel count. | Must be an integer greater than or equal to `2`; otherwise `ValueError`. |
| `save_dir` | `str` | `"./checkpoints/llie/LLFlow"` | Default checkpoint/config output directory inherited from `LLVModel`. | Not otherwise validated. |
| `condition_channels` | `int` | `32` | Condition-encoder feature width. | Converted with `int()` and must be positive; otherwise `ValueError`. |
| `condition_blocks` | `int` | `4` | Number of residual blocks in the condition encoder. | Converted with `int()` and must be positive. |
| `flow_layers` | `int` | `8` | Number of affine-coupling layers. | Converted with `int()` and must be positive. |
| `flow_hidden_channels` | `int` | `64` | Hidden width of each coupling network. | Converted with `int()` and must be positive. |
| `scale_clamp` | `float` | `2.0` | Tanh clamp applied to predicted log scale. | Converted with `float()` and must be positive. |
| `sample_temperature` | `float` | `0.0` | Standard deviation multiplier for inference latent sampling. | Converted with `float()` and must be non-negative. |
| `mode` | `str` | `"inference"` | Selects the tensor inference result or structured training result. | Exactly `"train"` or `"inference"`; otherwise `ValueError`. |

Every configuration key can also be supplied directly through constructor `**kwargs`; it overrides the same key in `config`.

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LLFlow",
    "input.jpg",
    output="results/llflow/output.png",
    config={"sample_temperature": 0.1},
)
```

```python
from openLLV.deepLearning.models.LLIE.LLFlow import LLFlow

model = LLFlow(condition_channels=48, flow_layers=6, mode="train")
```

## Checkpoint / Official Weights

Pass an openLLV `.pt`/`.pth` checkpoint as the first argument to `llv.predict`. A checkpoint stores the class name, merged configuration, and state dictionary. This implementation does not automatically download official weights.
