# LEDNet

> Task: low-light image enhancement and deblurring (LLIE)

LEDNet is openLLV's encoder-decoder combining pyramid pooling, curve attention, and dynamic filtering.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/pdf/2202.03373 |
| Official source code | https://github.com/sczhou/LEDNet |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/LEDNet.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/LEDNet.py` |
| Class name | `LEDNet` |
| Registered name | `LEDNet` (no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/LEDNet_Loss.py` |

## Implementation Notes

Three downsampling stages feed pyramid-pooling and curve-attention modules; the decoder applies generated dynamic kernels and optional skip additions. Inference returns the tensor. Training returns standardized output, with `aux.side_output` only when side supervision is active. Forward `side_loss=None` enables it only when both configured `use_side_loss` and training `mode` are true; an explicit boolean overrides that decision. The YAML overrides direct-construction `use_side_loss` from `False` to `True` and omits architecture keys that retain source defaults.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration mapping; a non-dictionary raises `TypeError`; `**kwargs` override it. |
| `model_name` | `str` | `"LEDNet"` | Base-class metadata. |
| `input_channels` | `int` | `3` | Input/output channels; must be a positive integer. |
| `save_dir` | `str` | `"./checkpoints/llie/LEDNet"` | Default checkpoint directory. |
| `channels` | `List[int]` | `[32, 64, 128, 128]` | Four stage widths; must contain four positive values. |
| `connection` | `bool` | `False` | Enables decoder-to-encoder skip additions. Compatible stage shapes/channels are required but not explicitly validated. |
| `use_side_loss` | `bool` | `False` | Default side-supervision switch. The YAML overrides it to `True`. |
| `mode` | `str` | `"inference"` | Must be `"train"` or `"inference"`. |
| `kernel_size` | `int` | `5` | Dynamic convolution kernel size; must be odd. Positivity is not explicitly validated. |
| `curve_n` | `int` | `3` | Curve-attention iteration count; must be positive. |
| `ppm_bins` | `Tuple[int, ...]` | `(1, 2, 3, 6)` | Adaptive-pooling output sizes for pyramid pooling; not explicitly validated. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LEDNet", "input.jpg", output="results/lednet/output.png",
    config={"connection": True, "curve_n": 4},
)
```

## Checkpoint / Official Weights

Use an openLLV checkpoint path as the prediction target. No automatic official-weight download is implemented.
