# Zero-IG

> Task: low-light image enhancement (LLIE)

Zero-IG combines illumination-guided enhancement with two denoising networks and is registered in openLLV as `ZeroIG`.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf |
| Official source code | https://github.com/Doyle59217/ZeroIG |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/ZeroIG.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/ZeroIG.py` |
| Class name | `ZeroIG` |
| Registered name | `ZeroIG` (no aliases; lookup is case-insensitive and trims surrounding whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `ZeroIG_Loss` in `openLLV/deepLearning/loss/LLIELoss/ZeroIG_Loss.py` (registered as `zeroig`; aliases `zeroig_loss`, `zero_ig`, `zero-ig`, `ZeroIG_Loss`) |

## Implementation Notes

`ZeroIG(config=None, **kwargs)` merges keyword overrides after `config`. Training initializes weights and returns a standardized dictionary whose `aux.outputs` is the 21-tensor tuple consumed by `ZeroIG_Loss`; inference returns the final three-channel enhanced/denoised tensor. Finetuning returns a standardized dictionary containing `H2` and `H3`. In `finetune` mode, a truthy `pretrained_weights` path is loaded immediately with `torch.load`; only keys present in the current state dictionary are copied, and load failures are printed and suppressed. `set_mode()` changes forward routing after construction, but it does not initialize weights or load finetune weights. The subnetworks are hard-coded around three-channel RGB images despite the shared `input_channels` metadata.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Dictionary merged over defaults. | Non-dictionary, non-`None` values raise `TypeError`; unknown keys are retained but ignored unless consumed. |
| `**kwargs` | `Any` | `{}` | Configuration overrides merged after `config`. | Same key semantics as `config`. |
| `model_name` | `str` | `"ZeroIG"` | Shared model metadata. | Not validated or used to construct the architecture. |
| `input_channels` | `int` | `3` | Shared input-channel metadata. | Must be a positive integer; model layers still assume three-channel input. |
| `save_dir` | `str` | `"./checkpoints/llie/ZeroIG"` | Default checkpoint/configuration output directory. | No constructor validation. |
| `enhance_layers` | `int` | `3` | Number of residual enhancement blocks. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `enhance_channels` | `int` | `64` | Enhancement-network feature width. | No explicit validation; must be accepted by PyTorch layer constructors. |
| `denoise1_channels` | `int` | `48` | First denoiser feature width. | No explicit validation; must be accepted by PyTorch layer constructors. |
| `denoise2_channels` | `int` | `48` | Second denoiser feature width. | No explicit validation; must be accepted by PyTorch layer constructors. |
| `mode` | `str` | `"inference"` | Selects training, inference, or finetuning behavior and output. | Exactly `"train"`, `"inference"`, or `"finetune"`; otherwise `ValueError` is raised. |
| `pretrained_weights` | `Optional[str]` | `None` | State-dictionary path loaded during construction in finetune mode. | Loading occurs only when `mode="finetune"` and the value is truthy; errors are printed and suppressed. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroIG",
    "input.jpg",
    output="results/zero-ig/output.png",
    config={"enhance_layers": 4, "mode": "inference"},
)
```

```python
import openLLV as llv

result = llv.train("ZeroIG", model_params={"mode": "train"})
```

## Checkpoint / Official Weights

For normal openLLV checkpoints, pass a `.pt` or `.pth` path to `llv.predict`. `pretrained_weights` is a separate finetuning path that expects a raw mapping of state-dictionary keys to tensors, not an openLLV checkpoint wrapper. No automatic download is implemented.
