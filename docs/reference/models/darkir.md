# DarkIR

> Task: low-light image restoration (LLIE)

DarkIR is openLLV's encoder-decoder model for joint low-light enhancement and restoration.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2025/papers/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.pdf |
| Official source code | https://github.com/cidautai/DarkIR |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/DarkIR.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/DarkIR.py` |
| Class name | `DarkIR` |
| Registered name | `DarkIR` (no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/DarkIR_Loss.py` |

## Implementation Notes

DarkIR zero-pads spatial dimensions to `2 ** len(enc_blk_nums)` and crops the main residual output back to the input size. Training mode returns the standardized output dictionary; `aux.side_output` is present only when side loss is enabled, and remains at bottleneck resolution. Passing forward `side_loss=True` persistently sets `model.config["side_loss"] = True`. The YAML sets `side_loss: true`, while direct construction defaults to `False`.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration mapping; a non-dictionary raises `TypeError`; `**kwargs` override it. |
| `model_name` | `str` | `"DarkIR"` | Base-class metadata. |
| `input_channels` | `int` | `3` | Image channels; must be a positive integer. |
| `save_dir` | `str` | `"./checkpoints/llie/DarkIR"` | Default checkpoint directory. |
| `width` | `int` | `32` | Initial feature width; must be positive. |
| `middle_blk_num_enc` | `int` | `2` | Encoder-middle block count; must be positive. |
| `middle_blk_num_dec` | `int` | `2` | Decoder-middle block count; must be positive. |
| `enc_blk_nums` | `List[int]` | `[1, 2, 3]` | Block count at each encoder level; determines padding multiple. No explicit element validation. |
| `dec_blk_nums` | `List[int]` | `[3, 1, 1]` | Block count at each decoder level. It should match encoder depth for complete skip/upsampling traversal; not explicitly validated. |
| `dilations` | `List[int]` | `[1, 4, 9]` | Dilations supplied to every decoder block; not explicitly validated. |
| `extra_depth_wise` | `bool` | `True` | Enables the extra depthwise branch in encoder/decoder blocks. |
| `mode` | `str` | `"inference"` | Must be `"train"` or `"inference"`. |
| `side_loss` | `bool` | `False` | Enables bottleneck side output. The YAML overrides this to `True`. Forward `side_loss=True` also enables it persistently. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "DarkIR", "input.jpg", output="results/darkir/output.png",
    config={"width": 32, "extra_depth_wise": True},
)
```

## Checkpoint / Official Weights

Use an openLLV checkpoint path as the prediction target. No automatic official-weight download is implemented.
