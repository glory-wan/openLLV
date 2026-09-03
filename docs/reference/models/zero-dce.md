# Zero-DCE

> Task: low-light image enhancement (LLIE)

Zero-DCE estimates per-pixel curve parameters and applies eight quadratic enhancement steps; its openLLV registered name is `ZeroDCE`.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf |
| Official source code | https://github.com/Li-Chongyi/Zero-DCE |
| Official project page | https://li-chongyi.github.io/Proj_Zero-DCE.html |
| Default configuration | `openLLV/deepLearning/config/ZeroDCE.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/ZeroDCE.py` |
| Class name | `ZeroDCE` |
| Registered name | `ZeroDCE` (no aliases; lookup is case-insensitive and trims surrounding whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `ZeroDCE_Loss` in `openLLV/deepLearning/loss/LLIELoss/ZeroDCE_Loss.py` (registered as `zerodce`; no aliases) |

## Implementation Notes

`ZeroDCE(config=None, **kwargs)` merges keyword overrides after `config`. Seven convolution layers estimate 24 curve-map channels, which `forward` splits into eight three-channel maps and applies in eight curve updates. All convolution weights are initialized from the normal distribution $N(0, 0.02)$. The reference-free training loss computes exposure control from the enhanced image. Training mode returns a standardized dictionary with `pred`, `aux.enhanced`, and `aux.r`; inference returns the enhanced tensor.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Dictionary merged over defaults. | Non-dictionary, non-`None` values raise `TypeError`; unknown keys are retained but ignored unless consumed. |
| `**kwargs` | `Any` | `{}` | Configuration overrides merged after `config`. | Same key semantics as `config`. |
| `model_name` | `str` | `"ZeroDCE"` | Shared model metadata. | Not validated or used to construct the architecture. |
| `input_channels` | `int` | `3` | Input channels of the first convolution. | Must be a positive integer. In practice it must be `3`, because the curve maps and update arithmetic are three-channel. |
| `save_dir` | `str` | `"./checkpoints/llie/ZeroDCE"` | Default checkpoint/configuration output directory. | No constructor validation. |
| `number_f` | `int` | `32` | Feature width of the curve-estimation network. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `mode` | `str` | `"inference"` | Selects training dictionary output or inference tensor output. | Exactly `"train"` or `"inference"`; otherwise `ValueError` is raised. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/zero-dce/output.png",
    config={"number_f": 24},
)
```

```python
import openLLV as llv

result = llv.train("ZeroDCE", root_dir="data/LOL-v1")
```

## Checkpoint / Official Weights

Pass an openLLV `.pt` or `.pth` checkpoint to `llv.predict`; explicit predictor configuration overrides the saved configuration. No official-weight downloader is implemented.
