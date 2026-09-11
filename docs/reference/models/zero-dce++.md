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

`ZeroDCEPlusPlus(config=None, **kwargs)` merges keyword overrides after `config`. Seven depthwise-separable blocks estimate one three-channel curve map, which is applied in eight quadratic updates. The default `scale_factor` is `12` in both training and inference. When it is not `1`, the input is resized by `1 / scale_factor` for curve estimation. The curve map is then resized directly to the original input dimensions using bilinear interpolation with `align_corners=True`, preserving the official upsampling alignment while supporting dimensions not divisible by the factor (including 512 with factor 12). The image is not cropped. Optional convolution-weight initialization uses $N(0, 0.02)$ and is disabled by default. The reference-free loss applies exposure control to the enhanced image with target `0.6`. Training returns a standardized dictionary with the enhanced result and curve map; inference returns a tensor.

The packaged `ZeroDCE++.yaml` sets `model.params.scale_factor: 12` and `data.resize: [512, 512]`, resizing training and validation images to 512×512 with the dataset's bilinear antialiased resize. Adam uses `lr: 0.0001` and `weight_decay: 0.0001`; `scheduler.name: null` keeps the learning rate constant, and `train.grad_clip: 0.1` clips the global gradient norm. These defaults remain overridable. The training scale factor is therefore 12, whereas the official training script defaults to 1.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Dictionary merged over defaults. | Non-dictionary, non-`None` values raise `TypeError`; unknown keys are retained but ignored unless consumed. |
| `**kwargs` | `Any` | `{}` | Configuration overrides merged after `config`. | Same key semantics as `config`. |
| `model_name` | `str` | `"ZeroDCEPlusPlus"` | Shared model metadata. | Not validated or used to construct the architecture. |
| `input_channels` | `int` | `3` | Channels accepted by the first depthwise block. | Must be a positive integer; operationally must be `3` because the curve output and image arithmetic are three-channel. |
| `save_dir` | `str` | `"./checkpoints/llie/ZeroDCEPlusPlus"` | Default checkpoint/configuration output directory. | No constructor validation. |
| `number_f` | `int` | `32` | Feature width of the curve-estimation network. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `scale_factor` | `int \| float` | `12` | Input downsampling factor used in both training and inference; the curve map is restored to the exact input size. | Must compare greater than `0`; otherwise `ValueError` is raised. Resized height and width must each be at least 1; input dimensions need not be divisible by the factor. |
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

result = llv.train("ZeroDCEPlusPlus", root_dir="datasets/my_dataset")
```

## Checkpoint / Official Weights

Pass an openLLV `.pt` or `.pth` checkpoint to `llv.predict`; explicit predictor configuration overrides saved values. Without an override, prediction preserves the model/checkpoint scale factor, including `1` in older checkpoints; `12` is the fallback when no value is saved or provided. No official-weight downloader is implemented.
