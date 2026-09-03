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

`ZeroDCEPlusPlus(config=None, **kwargs)` merges keyword overrides after `config`. Seven depthwise-separable blocks estimate one three-channel curve map, which is applied in eight quadratic updates. Following the official implementation, every mode resizes the input by `1 / scale_factor` when `scale_factor != 1`, estimates the curve map at that resolution, and resizes the curve map by `scale_factor` before enhancing the original image. Optional convolution-weight initialization uses $N(0, 0.02)$ and is disabled by default. The reference-free loss applies exposure control to the enhanced image with target `0.6`. Training returns a standardized dictionary with the enhanced result and curve map; inference returns a tensor.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Dictionary merged over defaults. | Non-dictionary, non-`None` values raise `TypeError`; unknown keys are retained but ignored unless consumed. |
| `**kwargs` | `Any` | `{}` | Configuration overrides merged after `config`. | Same key semantics as `config`. |
| `model_name` | `str` | `"ZeroDCEPlusPlus"` | Shared model metadata. | Not validated or used to construct the architecture. |
| `input_channels` | `int` | `3` | Channels accepted by the first depthwise block. | Must be a positive integer; operationally must be `3` because the curve output and image arithmetic are three-channel. |
| `save_dir` | `str` | `"./checkpoints/llie/ZeroDCEPlusPlus"` | Default checkpoint/configuration output directory. | No constructor validation. |
| `number_f` | `int` | `32` | Feature width of the curve-estimation network. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `scale_factor` | `int | float` | `1` | Reciprocal input-resize and curve-map-resize factor used in both training and inference. | Must compare greater than `0`; otherwise `ValueError` is raised. When not `1`, input dimensions must round-trip through both scaling operations to match the original image. |
| `initialize_weights` | `bool` | `False` | Whether to initialize every convolution weight from $N(0, 0.02)$ during construction. | Must be a Boolean; otherwise `TypeError` is raised. Biases retain their PyTorch initialization. |
| `mode` | `str` | `"inference"` | Selects the training dictionary or inference tensor output contract. | Exactly `"train"` or `"inference"`; otherwise `ValueError` is raised. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroDCEPlusPlus",
    "input.jpg",
    output="results/zero-dce++/output.png",
    config={
        "scale_factor": 2,
        "initialize_weights": True,
        "mode": "inference",
    },
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
