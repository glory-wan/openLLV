# KinD++

> Task: low-light image enhancement (LLIE)

KinD++ is openLLV's enhanced Retinex decomposition/restoration/illumination-adjustment model.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1007/s11263-020-01407-x |
| Official source code | https://github.com/zhangyhuaee/KinD_plus |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/KinD++.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/KinDPlusPlus.py` |
| Class name | `KinDPlusPlus` |
| Registered name | `KinDPlusPlus` (alias: `Kind++`; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/KinDPlusPlus_Loss.py` |

## Implementation Notes

KinD++ decomposes the input, restores reflectance with its multi-scale illumination-aware network, adjusts illumination using a constant ratio map, multiplies both results, and clamps to `[0, 1]`. Inference returns the tensor; training returns standardized output with all intermediate tensors and `decompose_fn`. The YAML agrees with model defaults except that it omits `mode`, `model_name`, and `save_dir`.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration mapping; a non-dictionary raises `TypeError`; `**kwargs` override it. |
| `model_name` | `str` | `"KinDPlusPlus"` | Base-class metadata. |
| `input_channels` | `int` | `3` | Image channels; must be a positive integer. |
| `save_dir` | `str` | `"./checkpoints/llie/KinDPlusPlus"` | Default checkpoint directory. |
| `decomposition_channels` | `int` | `32` | Decomposition-network base width; converted to `int` and must be positive. |
| `restoration_channels` | `int` | `32` | Restoration-network base width; converted to `int` and must be positive. |
| `adjustment_channels` | `int` | `32` | Illumination-adjustment width; converted to `int` and must be positive. |
| `illumination_ratio` | `float` | `5.0` | Constant exposure-ratio map value; must be positive. |
| `mode` | `str` | `"inference"` | Must be `"train"` or `"inference"`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "Kind++", "input.jpg", output="results/kind++/output.png",
    config={"illumination_ratio": 4.0},
)
```

## Checkpoint / Official Weights

Use an openLLV checkpoint path as the prediction target. No automatic official-weight download is implemented.
