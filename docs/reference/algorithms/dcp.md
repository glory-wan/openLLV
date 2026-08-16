# DarkChannel (DCP)

> Documentation group: traditional dehazing / low-light enhancement

DarkChannel applies the Dark Channel Prior to the inverted input and inverts the recovered image back, producing a low-light enhancement result.

## Links

| Type | URL |
| --- | --- |
| Paper | https://ieeexplore.ieee.org/document/5206515 |
| Official source code | None |
| Official project page | None |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/tradition/algorithms/Dehazing/DCP.py` |
| Class name | `DarkChannel` |
| Registered name | `DarkChannel` (alias: `dcp`; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

Public three-channel input and NumPy output use RGB. The base class converts it to the three-channel BGR, 8-bit working image expected by `_enhance()` and converts the result back to RGB. DCP estimates atmospheric light and transmission on the inverted image, refines transmission with a grayscale guided filter, and returns values scaled by `255`. Unlike the other algorithms in this group, `_enhance()` ignores all runtime keyword arguments: change algorithm values at construction or with `set_params()` instead. Factory creation filters unsupported constructor keys and emits `UserWarning`.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `size` | `int` | `15` | Rectangular dark-channel erosion kernel size. | Must be `> 0`. |
| `omega` | `float` | `0.95` | Transmission-estimation weight. | Must be in `(0, 1]`. |
| `t_min` | `float` | `0.1` | Transmission floor during recovery. | Must be in `(0, 1)`. |
| `guided_radius` | `int` | `60` | Guided-filter box size. | Must be `> 0`. |
| `guided_eps` | `float` | `0.0001` | Guided-filter regularization. | Must be `> 0`. |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | Base output format. | Other values raise `ValueError`. |
| `keep_dtype` | `bool` | `True` | Cast output back to input dtype. | Non-`bool` raises `TypeError`. |
| `clip_output` | `bool` | `True` | Clip output to the resolved input value range. | Non-`bool` raises `TypeError`. |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | Input value range; auto distinguishes ordinary float `[0,1]` and `[0,255]` inputs. | Use `"byte"` when a byte-range float image has maximum `<= 1`. Custom bounds must be finite and increasing; out-of-range values raise `ValueError`. |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("dcp", "input.jpg", output="results/dcp/output.png", size=9, omega=0.9)
```

```python
from openLLV.tradition.algorithms.Dehazing.DCP import DarkChannel

enhancer = DarkChannel(t_min=0.15)
enhancer.set_params(guided_radius=31)
result = enhancer("input.jpg")
```
