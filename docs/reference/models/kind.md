# KinD

> Task: low-light image enhancement (LLIE)

KinD is openLLV's Retinex-style decomposition, reflectance-restoration, and illumination-adjustment model.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1145/3343031.3350926 |
| Official source code | https://github.com/zhangyhuaee/KinD |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/KinD.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/KinD.py` |
| Class name | `KinD` |
| Registered name | `KinD` (no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/KinD_Loss.py` |

## Implementation Notes

KinD decomposes the input into reflectance and illumination, restores reflectance, adjusts illumination with a constant `illumination_ratio` map, multiplies them, and clamps to `[0, 1]`. Inference returns the tensor. Training returns the standardized dictionary with all intermediate tensors and `decompose_fn`. The YAML agrees with model defaults except that it omits `mode`, `model_name`, and `save_dir`.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration mapping; a non-dictionary raises `TypeError`; `**kwargs` override it. |
| `model_name` | `str` | `"KinD"` | Base-class metadata. |
| `input_channels` | `int` | `3` | Image channels; must be a positive integer. |
| `save_dir` | `str` | `"./checkpoints/llie/KinD"` | Default checkpoint directory. |
| `decomposition_channels` | `int` | `64` | Decomposition-network width; converted to `int` and must be positive. |
| `decomposition_layers` | `int` | `5` | Decomposition-network layer count; converted to `int` and must be positive. |
| `restoration_channels` | `int` | `32` | Restoration-network width; converted to `int` and must be positive. |
| `adjustment_channels` | `int` | `32` | Illumination-adjustment width; converted to `int` and must be positive. |
| `adjustment_layers` | `int` | `3` | Illumination-adjustment layer count; converted to `int` and must be positive. |
| `illumination_ratio` | `float` | `5.0` | Constant exposure-ratio map value; must be positive. |
| `mode` | `str` | `"inference"` | Must be `"train"` or `"inference"`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "KinD", "input.jpg", output="results/kind/output.png",
    config={"illumination_ratio": 4.0},
)
```

## Checkpoint / Official Weights

Use an openLLV checkpoint path as the prediction target. No automatic official-weight download is implemented.
