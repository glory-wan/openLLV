# AHE

> Documentation group: base methods

AHE approximates adaptive histogram equalization with OpenCV CLAHE at a fixed `clipLimit=255.0`.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1016/S0734-189X(87)80186-X |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/BaseMethods/AHE.py` |
| Class name | `AHE` |
| Registered name | `ahe` (class name also registers; no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

Public three-channel input and NumPy output use RGB. `LLVEnhancer` converts RGB to internal BGR before `_enhance()` and converts the result back to RGB afterward. Grayscale input is equalized directly. Internally, `rgb` equalizes every BGR channel; other spaces equalize V, L, Y, or L respectively and convert back to BGR. Non-`uint8` data is converted to `uint8` first. Per-call algorithm kwargs are ignored. `set_params` changes attributes but does not rebuild the cached OpenCV CLAHE object, so construct a new instance to change `tile_grid_size` effectively.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | One of `rgb`, `bgr` (alias of `rgb`), `hsv`, `hls`, `yuv`, `ycbcr` (alias of `yuv`), or `lab`; normalized case-insensitively after trimming. Non-string raises `TypeError`; unsupported value raises `ValueError`. |
| `tile_grid_size` | `Tuple[int, int]` | `(8, 8)` | CLAHE grid; tuple/list of exactly two positive non-boolean integers, otherwise `ValueError`. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format; unsupported value raises `ValueError`. Traditional `Predictor` forces instances to `"numpy"`. |
| `keep_dtype` | `bool` | `True` | Cast result to input dtype; non-boolean raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip to the resolved input value range before casting; non-boolean raises `TypeError`. |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | Input value range. Auto infers float `[0,1]` or `[0,255]`; use `"byte"` for ambiguous dark byte-range floats whose maximum is `<= 1`. Custom bounds must be finite and increasing. Values outside the selected range raise `ValueError`. |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("ahe", "input.jpg", output="results/ahe.png", color_space="lab", tile_grid_size=(4, 4))
```
