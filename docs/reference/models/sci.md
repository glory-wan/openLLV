# SCI

> Task: low-light image enhancement (LLIE)

SCI is a staged illumination-estimation and calibration model. In openLLV it is a deep-learning component selected with the registered name `SCI`.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| Official source code | https://github.com/vis-opt-group/SCI |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/SCI.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/SCI.py` |
| Class name | `SCI` |
| Registered name | `SCI` (no aliases; lookup is case-insensitive and trims surrounding whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `Sci_Loss` in `openLLV/deepLearning/loss/LLIELoss/Sci_Loss.py` (registered as `sci`; alias `sci_loss`) |

## Implementation Notes

The constructor is `SCI(config=None, **kwargs)`. `LLVModel` merges shared defaults, `config`, then `kwargs`, so a keyword override wins over the same key in `config`. The enhancer estimates an illumination map and each training stage calibrates the next input. With `mode="train"`, `forward(x)` returns a standardized dictionary whose `pred` is the final reflectance and whose `aux` contains `enhanced`, `ilist`, `rlist`, `inlist`, and `attlist`. With `mode="inference"`, only the first-stage enhanced tensor is returned. The network layers themselves are hard-coded for three-channel input; changing `input_channels` does not change them.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration dictionary merged over all defaults. | A non-dictionary, non-`None` value raises `TypeError`. Unknown keys are retained but otherwise ignored unless consumed by model code. |
| `**kwargs` | `Any` | `{}` | Flat configuration overrides merged after `config`. | Same key semantics as `config`. |
| `model_name` | `str` | `"SCI"` | Shared `LLVModel` metadata stored in `model.config`. | Not validated or used to construct the architecture. |
| `input_channels` | `int` | `3` | Shared input-channel metadata. | Must be a positive integer or `ValueError` is raised. SCI layers still require three-channel tensors. |
| `save_dir` | `str` | `"./checkpoints/llie/SCI"` | Default directory used by `save_model()` and `save_config()`. | No constructor validation. |
| `stage` | `int` | `3` | Number of staged enhancement/calibration iterations in training. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `enhance_layers` | `int` | `1` | Number of residual blocks in the enhancement network. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `enhance_channels` | `int` | `3` | Feature width of the enhancement network. | No explicit validation; must be acceptable to PyTorch convolution and batch-normalization constructors. |
| `calibrate_layers` | `int` | `3` | Number of residual blocks in the calibration network. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `calibrate_channels` | `int` | `16` | Feature width of the calibration network. | No explicit validation; must be acceptable to PyTorch convolution and batch-normalization constructors. |
| `mode` | `str` | `"inference"` | Selects the forward-output contract. | Exactly `"train"` or `"inference"`; otherwise `ValueError` is raised. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "SCI",
    "input.jpg",
    output="results/sci/output.png",
    config={"stage": 2, "mode": "inference"},
)
```

```python
import openLLV as llv

result = llv.train("SCI", model_params={"stage": 3, "mode": "train"})
```

## Checkpoint / Official Weights

Pass an openLLV `.pt` or `.pth` checkpoint as the prediction method. Checkpoint loading restores the saved model class and configuration; an explicit predictor `config` overrides saved configuration values. This implementation does not automatically download official SCI weights.
