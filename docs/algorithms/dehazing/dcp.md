# DCP / Dark Channel

> Documentation group: Dehazing

DCP is a traditional enhancement algorithm based on the dark channel prior already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://ieeexplore.ieee.org/document/5206515 |
| Extended journal version | https://pubmed.ncbi.nlm.nih.gov/20820075/ |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

DCP, or Dark Channel Prior, was originally used for single-image dehazing. Its core observation is that in most local regions of non-sky natural images, at least one color channel has very low pixel values. This prior can be used to estimate transmission and atmospheric light.

In low-light enhancement, a common practice is to invert the low-light image, convert it into a haze-like form, apply the dark channel prior for recovery, and finally invert it back to obtain the enhanced image.

## Location in openLLV

| Item | Location |
| --- | --- |
| Algorithm implementation | `openLLV/tradition/algorithms/Dehazing/DCP.py` |
| Algorithm class name | `DarkChannel` |
| Registered name | `DarkChannel` |
| Alias | `dcp` |
| Base class | `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py` |

## Implementation Notes

openLLV's DCP pipeline includes:

1. Normalize the input image.
2. Invert the image.
3. Compute the dark channel.
4. Estimate atmospheric light.
5. Estimate and refine the transmission map.
6. Recover the image and invert it again.

Main parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `size` | `int` | `15` | Dark-channel erosion kernel size |
| `omega` | `float` | `0.95` | Transmission estimation weight |
| `t_min` | `float` | `0.1` | Minimum transmission |
| `guided_radius` | `int` | `60` | Guided filter radius |
| `guided_eps` | `float` | `1e-4` | Guided filter regularization term |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "dcp",
    "input.jpg",
    output="results/dcp/output.jpg",
    size=15,
    omega=0.95,
    t_min=0.1,
)
```

Folder batch processing:

```python
saved_paths = llv.predict(
    "dcp",
    "images/",
    output="results/dcp",
    progress_bar=True,
)
```
