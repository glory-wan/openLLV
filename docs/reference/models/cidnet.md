# CIDNet

> Task: low-light image enhancement (LLIE)

CIDNet is openLLV's HVI-space dual-branch enhancement model.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/abs/2502.20272 |
| Official source code | https://github.com/Fediory/HVI-CIDNet |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/CIDNet.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/CIDNet.py` |
| Class name | `CIDNet` |
| Registered name | `CIDNet` (alias: `HVI-CIDNet`; lookup is case-insensitive and trims surrounding whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/CIDNet_Loss.py` |

## Implementation Notes

The model converts RGB to HVI, processes hue/value and intensity in coupled encoder-decoder branches, and converts back to RGB. It replicate-pads height and width to multiples of 8 and crops the result to the original size. With `mode="inference"`, `forward` returns a tensor; with `mode="train"`, it returns `{"pred", "aux", "meta"}`, where `aux` contains `prediction_hvi` and `density_k`. The forward-only `input_gamma` keyword overrides the configured value for that call. The default YAML agrees with the model defaults except that it omits `mode`, `model_name`, and `save_dir`.

## Parameters

`config` and `**kwargs` are merged over these defaults; keyword values win. Unknown keys remain in `model.config` but are otherwise ignored unless consumed by another API.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration mapping; a non-dictionary raises `TypeError`. |
| `model_name` | `str` | `"CIDNet"` | Base-class metadata stored in `model.config`. |
| `input_channels` | `int` | `3` | Input channels. Must be exactly `3` for CIDNet. |
| `save_dir` | `str` | `"./checkpoints/llie/CIDNet"` | Default checkpoint directory used by `LLVModel.save_model`. |
| `channels` | `List[int]` | `[36, 36, 72, 144]` | Four stage widths. Both this list and `heads` must contain four values; all converted values must be positive. |
| `heads` | `List[int]` | `[1, 2, 4, 8]` | Four attention-head counts. Every channel count must be divisible by its paired head count. |
| `norm` | `bool` | `False` | Enables normalization in down/up-sampling blocks after boolean conversion. |
| `density_k` | `float` | `0.2` | Density parameter passed to `HVITransform`; converted to `float` and not explicitly range-validated here. |
| `input_gamma` | `float` | `1.0` | Gamma applied before HVI conversion. Must be positive at construction; may also be overridden per forward call through `model_kwargs`. |
| `saturation_scale` | `float` | `1.0` | Saturation scale passed to inverse HVI conversion; converted to `float`. |
| `intensity_scale` | `float` | `1.0` | Intensity scale passed to inverse HVI conversion; converted to `float`. |
| `clamp_output` | `bool` | `True` | If truthy, clamps the RGB result to `[0, 1]`. |
| `mode` | `str` | `"inference"` | Output contract; must be `"train"` or `"inference"`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "HVI-CIDNet",
    "input.jpg",
    output="results/cidnet/output.png",
    config={"density_k": 0.25, "saturation_scale": 0.9},
    model_kwargs={"input_gamma": 0.9},
)
```

## Checkpoint / Official Weights

Pass an openLLV `.pt` or `.pth` checkpoint as the first argument to `llv.predict`; a supplied `config` overrides its saved configuration. The repository does not automatically download official CIDNet weights.
