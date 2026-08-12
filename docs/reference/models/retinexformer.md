# RetinexFormer

> Task: low-light image enhancement (LLIE)

RetinexFormer is a staged Retinex-based transformer that combines illumination estimation with illumination-guided denoising.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.pdf |
| Official source code | https://github.com/caiyuanhao1998/Retinexformer |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/RetinexFormer.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/RetinexFormer.py` |
| Class name | `RetinexFormer` |
| Registered name | `RetinexFormer` (no aliases) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/RetinexFormer_Loss.py` (`retinexformer`; aliases: `retinexformer_loss`, `RetinexFormer_Loss`, `retinexformer_l1`) |

## Implementation Notes

Each stage estimates illumination features and a map, forms `image * illumination_map + image`, and applies an illumination-guided U-Net-style denoiser. Spatial dimensions are reflect-padded to a multiple of `2 ** levels` and cropped back. Multiple stages consume the previous stage's output. Training returns `{"pred", "aux", "meta"}` with every stage's illumination intermediates; inference returns a tensor. Output clamping occurs before either return mode when enabled.

`num_blocks` is normalized inside the denoiser: values are integer-converted, a short sequence repeats its final element until it has `levels + 1` entries, and a longer sequence is truncated. An empty sequence fails with `IndexError`; no explicit positivity validation is performed. The constructor merges defaults, `config`, then `**kwargs`.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration dictionary. | Must be a dictionary or `None`. |
| `model_name` | `str` | `"RetinexFormer"` | Stored model name inherited from `LLVModel`. | Not otherwise validated. |
| `input_channels` | `int` | `3` | Input channel count. | Base validation requires a positive integer. Architecture composition practically requires it to match `output_channels` between stages and for the residual additions. |
| `save_dir` | `str` | `"./checkpoints/llie/RetinexFormer"` | Default checkpoint/config directory. | Not otherwise validated. |
| `output_channels` | `int` | `3` | Restored-image channel count. | Compared to zero and must be positive; non-numeric incompatible values may raise `TypeError`. |
| `feature_channels` | `int` | `32` | Base feature width and attention head dimension. | Compared to zero and must be positive; converted with `int()` during construction. |
| `stage` | `int` | `1` | Number of sequential RetinexFormer stages. | Compared to zero and must be positive; converted with `int()` for construction. |
| `levels` | `int` | `2` | Encoder/decoder depth; padding factor is `2 ** levels`. | Compared to zero and must be positive; converted with `int()`. |
| `num_blocks` | `Sequence[int]` | `[1, 1, 1]` | Attention-block counts for encoder levels and bottleneck. | Must be a non-empty iterable of integer-convertible values. It is extended/truncated to `levels + 1`; no positive-value check exists. |
| `mode` | `str` | `"inference"` | Selects structured training output or tensor inference output. | Exactly `"train"` or `"inference"`. |
| `clamp_output` | `bool` | `True` | Clamps final output to `[0, 1]`. | Read with `config.get(..., True)` and truth-tested in both modes. |

Configuration keys passed as constructor `**kwargs` override `config`; unknown keys remain stored but unused. Forward `**kwargs` are accepted and ignored.

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "RetinexFormer",
    "input.jpg",
    output="results/retinexformer/output.png",
    config={"stage": 2, "num_blocks": [1, 2, 2]},
)
```

```python
from openLLV.deepLearning.models.LLIE.RetinexFormer import RetinexFormer

model = RetinexFormer(feature_channels=24, levels=2, clamp_output=False)
```

## Checkpoint / Official Weights

Pass an openLLV checkpoint directly to `llv.predict`. This implementation does not automatically fetch or remap official RetinexFormer checkpoints.
