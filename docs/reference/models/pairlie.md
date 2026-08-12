# PairLIE

> Task: low-light image enhancement (LLIE)

PairLIE is a Retinex-style decomposition model trained from paired low-light instances and usable for single-image inference.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf |
| Official source code | https://github.com/zhenqifu/PairLIE |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/PairLIE.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/PairLIE.py` |
| Class name | `PairLIE` |
| Registered name | `PairLIE` (alias: `Pair-LIE`) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/PairLIE_Loss.py` (`pairlie`; aliases: `pairlie_loss`, `PairLIE-Loss`) |

## Implementation Notes

Three five-layer convolutional estimators predict a denoised image, a one-channel illumination map, and a three-channel reflectance map. Enhancement computes `illumination ** enhancement_gamma * reflectance`. In inference mode, `forward(image)` returns only the prediction. Training mode returns `{"pred", "aux", "meta"}`; when `paired_image` is supplied, `aux` also contains its decomposition. `PairLIE_Loss` requires that paired decomposition, so unified training must provide a second low-light instance. The class advertises this through `requires_paired_forward=True`.

The estimator module names match the official implementation for raw state-dictionary compatibility. The constructor is `PairLIE(config=None, **kwargs)`; keyword values override `config`.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration dictionary. | Must be a dictionary or `None`. |
| `model_name` | `str` | `"PairLIE"` | Stored model name inherited from `LLVModel`. | Not otherwise validated. |
| `input_channels` | `int` | `3` | Input channel count. | Must be exactly `3`; otherwise `ValueError`. The internal estimators are fixed to RGB. |
| `save_dir` | `str` | `"./checkpoints/llie/PairLIE"` | Default checkpoint/config directory. | Not otherwise validated. |
| `feature_channels` | `int` | `64` | Hidden width shared by the three estimators. | Converted with `int()` and must be positive. |
| `enhancement_gamma` | `float` | `0.2` | Exponent applied to the illumination map during composition. | Converted with `float()` and must be positive. |
| `clamp_output` | `bool` | `True` | Clamps composed predictions to `[0, 1]`. | Truth-tested; no strict type check. |
| `mode` | `str` | `"inference"` | Selects single prediction or structured training output. | Exactly `"train"` or `"inference"`. |
| `forward(..., paired_image=...)` | `Optional[torch.Tensor]` | `None` | Second low-light instance of the same scene for training decomposition. | Optional at model level, but `PairLIE_Loss` raises `ValueError` when the paired target is absent and `KeyError` when paired decomposition fields are missing. |

All configuration keys may be passed through constructor `**kwargs`; unknown keys are retained but unused. Other forward `**kwargs` are accepted and ignored.

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "Pair-LIE",
    "input.jpg",
    output="results/pairlie/output.png",
    config={"enhancement_gamma": 0.25, "clamp_output": True},
)
```

```python
from openLLV.deepLearning.models.LLIE.PairLIE import PairLIE

model = PairLIE(mode="train")
training_output = model(first_low_light_tensor, paired_image=second_low_light_tensor)
```

## Checkpoint / Official Weights

Pass an openLLV checkpoint directly to `llv.predict`. Although estimator names preserve official raw state-dictionary names, openLLV does not automatically download or load a raw official checkpoint.
