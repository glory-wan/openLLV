# Zero-IG

> Task: low-light image enhancement (`llie`)

Zero-IG is a zero-shot low-light enhancement and denoising model already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf |
| Official source code | https://github.com/Doyle59217/ZeroIG |

## Model Introduction

The full name of Zero-IG is Zero-shot Illumination-Guided joint denoising and adaptive enhancement. The method targets zero-shot scenarios and considers both low-light enhancement and noise suppression, using an illumination-guided mechanism to guide the enhancement and denoising process.

In openLLV, Zero-IG contains an enhancement network, a two-stage denoising network, a texture difference module, and local average pooling components. The configuration can set the number of enhancement layers, enhancement channels, denoising channels, and running mode.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/ZeroIG.py` |
| Model class name | `ZeroIG` |
| Default configuration | `openLLV/deepLearning/config/ZeroIG.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/ZeroIG_Loss.py` |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroIG",
    "input.jpg",
    output="results/ZeroIG/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    model="ZeroIG",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="zeroig",
    epochs=10,
    batch_size=4,
)
```
