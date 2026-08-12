# LLNet

> Task: low-light image enhancement and denoising (LLIE)

LLNet is an overlapping-patch, fully connected autoencoder registered for low-light enhancement and denoising.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1016/j.patcog.2016.06.008 |
| Official source code | https://github.com/kglore/llnet_color |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/LLNet.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/LLNet.py` |
| Class name | `LLNet` |
| Registered name | `LLNet` (no aliases) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/LLNet_Loss.py` (`llnet`; aliases: `llnet_loss`, `LLNet_Loss`, `sparse_denoising_autoencoder`) |

## Implementation Notes

LLNet reflect-pads the input, extracts overlapping patches, runs flattened patches through symmetric fully connected encoder/decoder stacks, folds the patches back, divides by overlap counts, and crops to the original size. Training mode returns `{"pred", "aux", "meta"}` with hidden activations and all regularized weights; inference returns the enhanced tensor. Large defaults (`hidden_dims=[2000, 1600, 1200]`) imply substantial memory use.

The constructor is `LLNet(config=None, **kwargs)`. Defaults, `config`, and keyword overrides are merged in that order.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration dictionary. | Must be a dictionary or `None`. |
| `model_name` | `str` | `"LLNet"` | Stored model name inherited from `LLVModel`. | Not otherwise validated. |
| `input_channels` | `int` | `3` | Input channels and patch-vector channel count. | Must be a positive integer. |
| `save_dir` | `str` | `"./checkpoints/llie/LLNet"` | Default checkpoint/config directory. | Not otherwise validated. |
| `patch_size` | `int` | `17` | Square patch side length. | Converted with `int()`; must be positive and odd. Reflect padding also requires a sufficiently large input at runtime. |
| `patch_stride` | `int` | `3` | Patch extraction/folding stride. | Converted with `int()` and must be positive. |
| `hidden_dims` | `Union[List[int], Tuple[int, ...]]` | `[2000, 1600, 1200]` | Encoder widths; the decoder reverses them. | Must be a non-empty list/tuple whose values convert to positive integers. |
| `activation` | `str` | `"sigmoid"` | Hidden-layer activation. | Exactly `"sigmoid"`, `"relu"`, or `"tanh"`. |
| `output_activation` | `str` | `"sigmoid"` | Final patch activation. | Exactly `"sigmoid"`, `"clamp"` (to `[0, 1]`), or `"none"`. |
| `mode` | `str` | `"inference"` | Selects structured training output or tensor inference output. | Exactly `"train"` or `"inference"`. |

All configuration keys may be provided through constructor `**kwargs` and override `config`; unknown keys remain stored but unused.

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LLNet",
    "input.jpg",
    output="results/llnet/output.png",
    config={"patch_stride": 2, "output_activation": "clamp"},
)
```

```python
from openLLV.deepLearning.models.LLIE.LLNet import LLNet

model = LLNet(patch_size=9, hidden_dims=[512, 256], activation="relu")
```

## Checkpoint / Official Weights

OpenLLV checkpoints can be passed directly to `llv.predict`. This implementation does not automatically download or convert weights from the official repository.
