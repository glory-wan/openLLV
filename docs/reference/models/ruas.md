# RUAS

> Task: unsupervised low-light image enhancement (LLIE)

RUAS is a Retinex-inspired unrolled model with searched illumination-estimation and denoising networks.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf |
| Official source code | https://github.com/KarelZhang/RUAS |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/RUAS.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/RUAS.py` |
| Class name | `RUAS` |
| Registered name | `RUAS` (no aliases) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/RUAS_Loss.py` (`ruas`; no aliases) |

## Implementation Notes

The enhancement network iteratively estimates clamped transmission maps and computes `input / transmission`; the denoising network subtracts predicted noise from the final enhanced image. Fixed `IEM_Genotype` and `NRM_Genotype` operation graphs define both searched networks. In training mode, `forward()` returns `{"pred", "aux", "meta"}` with all enhanced images, all transmission maps, and noise. Inference returns the final denoised image without an explicit final clamp.

If `pretrained_denoise_path` is truthy, initialization calls `torch.load(..., map_location="cpu")` and loads the value directly into `denoise_net`. Every exception is caught and printed, so a missing or incompatible file does not abort construction. The constructor is `RUAS(config=None, **kwargs)`; keyword values override `config`.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration dictionary. | Must be a dictionary or `None`. |
| `model_name` | `str` | `"RUAS"` | Stored model name inherited from `LLVModel`. | Not otherwise validated. |
| `input_channels` | `int` | `3` | Base-model input-channel metadata. | Must be a positive integer. The architecture itself is fixed to three-channel tensors; changing this key does not rebuild those fixed layers. |
| `save_dir` | `str` | `"./checkpoints/llie/RUAS"` | Default checkpoint/config directory. | Not otherwise validated. |
| `iem_nums` | `int` | `3` | Number of illumination-estimation iterations. | Compared directly to zero and must be positive; later consumed by `range()`, so an integer-compatible value is required. |
| `nrm_nums` | `int` | `3` | Number of denoising search blocks. | Compared directly to zero and must be positive; later consumed by `range()`. |
| `enhance_channel` | `int` | `3` | Feature width of the enhancement search blocks. | No explicit validation; must be a valid positive convolution channel count. With the fixed genotype's residual/identity operations, incompatible widths fail during construction or forward. |
| `denoise_channel` | `int` | `6` | Feature width of the denoising search blocks. | No explicit validation; must be a valid positive convolution channel count. |
| `mode` | `str` | `"inference"` | Selects structured training output or tensor inference output. | Exactly `"train"` or `"inference"`. |
| `pretrained_denoise_path` | `Optional[Union[str, pathlib.Path]]` | `None` | Raw denoising-network state-dictionary path loaded during construction. | Any truthy value triggers loading. Loading errors are caught and printed rather than raised. |

All configuration keys can be passed through constructor `**kwargs`; unknown keys remain stored but unused. Forward `**kwargs` are accepted and ignored.

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "RUAS",
    "input.jpg",
    output="results/ruas/output.png",
    config={"iem_nums": 4, "nrm_nums": 2},
)
```

```python
from openLLV.deepLearning.models.LLIE.RUAS import RUAS

model = RUAS(pretrained_denoise_path="weights/ruas_denoise.pt")
```

## Checkpoint / Official Weights

Use an openLLV full-model checkpoint by passing it to `llv.predict`. `pretrained_denoise_path` is different: it expects a state dictionary for `denoise_net` itself, not an openLLV checkpoint wrapper. Official weights are not downloaded automatically.
