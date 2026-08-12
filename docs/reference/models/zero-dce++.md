# Zero-DCE++

> Task: low-light image enhancement (LLIE)

Zero-DCE++ is a compact curve-estimation network based on depthwise-separable convolutions, registered as `ZeroDCEPlusPlus`.

## Links

| Type | URL |
| --- | --- |
| Paper | https://ieeexplore.ieee.org/document/9369102/ |
| Official source code | https://github.com/Li-Chongyi/Zero-DCE_extension |
| Official project page | https://li-chongyi.github.io/Proj_Zero-DCE++.html |
| Default configuration | `openLLV/deepLearning/config/ZeroDCE++.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/ZeroDCEPlusPlus.py` |
| Class name | `ZeroDCEPlusPlus` |
| Registered name | `ZeroDCEPlusPlus` (no aliases; lookup is case-insensitive and trims surrounding whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `ZeroDCE_extension_Loss` in `openLLV/deepLearning/loss/LLIELoss/ZeroDCE_Loss.py` (registered as `zerodce_extension`; aliases `zerodceplusplus`, `zerodce++`) |

## Implementation Notes

`ZeroDCEPlusPlus(config=None, **kwargs)` merges keyword overrides after `config`. Seven depthwise-separable blocks estimate one three-channel curve map, which is applied in eight quadratic updates. In inference only, `scale_factor > 1` downsamples the image before curve estimation and upsamples the curve map before enhancement. Training does not downsample, but still upsamples the curve map when `scale_factor > 1`; this makes its spatial size incompatible with the original input, so training configurations should use `scale_factor=1`. Training returns a standardized dictionary with the enhanced result and curve map; inference returns a tensor. `train_mode()` and `eval_mode()` update both the configured mode and the downsampling module.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Dictionary merged over defaults. | Non-dictionary, non-`None` values raise `TypeError`; unknown keys are retained but ignored unless consumed. |
| `**kwargs` | `Any` | `{}` | Configuration overrides merged after `config`. | Same key semantics as `config`. |
| `model_name` | `str` | `"ZeroDCEPlusPlus"` | Shared model metadata. | Not validated or used to construct the architecture. |
| `input_channels` | `int` | `3` | Channels accepted by the first depthwise block. | Must be a positive integer; operationally must be `3` because the curve output and image arithmetic are three-channel. |
| `save_dir` | `str` | `"./checkpoints/llie/ZeroDCEPlusPlus"` | Default checkpoint/configuration output directory. | No constructor validation. |
| `number_f` | `int` | `32` | Feature width of the curve-estimation network. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `scale_factor` | `int | float` | `1` | Inference curve-estimation downsampling and curve-map upsampling factor. | Must compare greater than `0`; otherwise `ValueError` is raised. Use `1` for training to preserve spatial compatibility. |
| `mode` | `str` | `"inference"` | Selects architecture routing and output contract. | Exactly `"train"` or `"inference"`; otherwise `ValueError` is raised. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroDCEPlusPlus",
    "input.jpg",
    output="results/zero-dce++/output.png",
    config={"scale_factor": 2, "mode": "inference"},
)
```

```python
import openLLV as llv

result = llv.train(
    "ZeroDCEPlusPlus",
    model_params={"mode": "train", "scale_factor": 1},
)
```

## Checkpoint / Official Weights

Pass an openLLV `.pt` or `.pth` checkpoint to `llv.predict`; explicit predictor configuration overrides saved values. No official-weight downloader is implemented.
