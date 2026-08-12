# LLFormer

> Task: low-light image enhancement (LLIE)

LLFormer is an axial-attention transformer model for full-resolution or tiled low-light enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/abs/2212.11548 |
| Official source code | https://github.com/TaoWangzj/LLFormer |
| Official project page | https://taowangzj.github.io/projects/LLFormer/ |
| Default configuration | `openLLV/deepLearning/config/LLFormer.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/LLFormer.py` |
| Class name | `LLFormer` |
| Registered name | `LLFormer` (alias: `LL-Former`) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/LLFormer_Loss.py` (`llformer`; aliases: `llformer_loss`, `LLFormer-Loss`) |

## Implementation Notes

LLFormer uses four resolution levels, axial attention, dual-gated feed-forward blocks, cross-layer fusion, and learned weighted skip connections. With `pad_input=True`, inputs are padded to multiples of 16 and cropped back afterward. Tiled inference is active only in inference mode when `tile_size` is not `None`; overlapping predictions are averaged. Training mode returns the standard `{"pred", "aux", "meta"}` dictionary and does not clamp the prediction. Inference returns a tensor, optionally clamped to `[0, 1]`.

The constructor is `LLFormer(config=None, **kwargs)`. Defaults, `config`, and `**kwargs` are merged in that order. The registry lookup is case-insensitive and trims surrounding whitespace.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration dictionary. | Must be a dictionary or `None`; otherwise `TypeError`. |
| `model_name` | `str` | `"LLFormer"` | Stored model name from `LLVModel`. | Not otherwise validated. |
| `input_channels` | `int` | `3` | Input channel count. | Must be a positive integer under base validation. |
| `save_dir` | `str` | `"./checkpoints/llie/LLFormer"` | Default checkpoint/config directory. | Not otherwise validated. |
| `output_channels` | `int` | `3` | Output channel count. | Converted with `int()` and must be positive. |
| `dim` | `int` | `16` | Base feature width. | Converted with `int()`; must be positive and even. |
| `num_blocks` | `List[int]` | `[2, 4, 8, 16]` | Transformer block counts for four levels. | Must be iterable with exactly four positive integer-convertible values. |
| `num_refinement_blocks` | `int` | `2` | Blocks in each refinement branch. | Converted with `int()` and must be positive. |
| `heads` | `List[int]` | `[1, 2, 4, 8]` | Attention head counts at four levels. | Exactly four positive integer-convertible values; each must divide the corresponding width `[dim, 2*dim, 4*dim, 8*dim]`. |
| `ffn_expansion_factor` | `float` | `2.66` | Feed-forward hidden-width multiplier. | Converted with `float()` and must be positive. |
| `bias` | `bool` | `False` | Enables biases in feed-forward and selected convolution layers. Axial attention itself always uses bias. | Converted with `bool()` during construction. |
| `layernorm_type` | `str` | `"WithBias"` | Layer-normalization implementation. | Exactly `"BiasFree"` or `"WithBias"`. |
| `attention` | `bool` | `True` | Controls whether learned fusion coefficients require gradients. | Converted with `bool()`; does not disable axial-attention modules. |
| `skip` | `bool` | `False` | Adds the input image to core output. | If truthy, input and output channel values must match; otherwise `ValueError`. |
| `pad_input` | `bool` | `True` | Pads spatial dimensions to multiples of 16 before inference/training. | Truth-tested; disabling it requires architecture-compatible input dimensions. |
| `clamp_output` | `bool` | `True` | Clamps inference output to `[0, 1]`. | Applied only in inference mode. |
| `tile_size` | `Optional[Union[int, Tuple[int, int]]]` | `None` | Default inference tile height/width; an integer means a square. | `None`, a positive integer, or two positive integer-convertible values; dimensions must be divisible by 16. |
| `tile_overlap` | `Union[int, Tuple[int, int]]` | `0` | Default tile overlap; an integer applies to both axes. | One or two non-negative integer-convertible values; each must be smaller than the matching tile size when tiling is configured at construction. |
| `mode` | `str` | `"inference"` | Selects structured training output or tensor inference output. | Exactly `"train"` or `"inference"`. |
| `forward(..., tile_size=...)` | `Optional[Union[int, Tuple[int, int]]]` | current `config["tile_size"]` | Per-call tiled-inference override, passed through `model_kwargs` in `llv.predict`. | Same normalization as `tile_size`; only used in inference mode. |
| `forward(..., tile_overlap=...)` | `Union[int, Tuple[int, int]]` | current `config["tile_overlap"]` | Per-call overlap override. | Same normalization as `tile_overlap`; runtime code requires valid strides but does not repeat the constructor's overlap-smaller-than-tile check. |

Configuration keys passed through constructor `**kwargs` override the same names in `config`. Other unknown keys are retained but unused.

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LL-Former",
    "input.jpg",
    output="results/llformer/output.png",
    config={"tile_size": 512, "tile_overlap": 64},
)
```

```python
import openLLV as llv

enhanced, _ = llv.predict(
    "LLFormer",
    "input.jpg",
    save=False,
    model_kwargs={"tile_size": (512, 768), "tile_overlap": (64, 64)},
)
```

## Checkpoint / Official Weights

An openLLV checkpoint can be passed directly to `llv.predict`. This implementation preserves the model architecture but does not download or remap official weights automatically.
