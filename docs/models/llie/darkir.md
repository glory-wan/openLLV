# DarkIR

> Task: low-light image enhancement (`llie`)

DarkIR is a robust low-light image restoration model already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/pdf/2412.13443 |
| CVPR paper page | https://openaccess.thecvf.com/content/CVPR2025/papers/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.pdf |
| Official source code | https://github.com/cidautai/DarkIR |

## Model Introduction

DarkIR targets robust low-light image restoration. It focuses not only on brightness improvement, but also on noise, color degradation, and detail loss commonly seen in low-light scenes. The model adopts an encoder-decoder structure and includes multi-scale or depthwise-separable related modules to improve dark-image restoration capability.

In openLLV, DarkIR's default configuration includes network width, the number of encoder/decoder blocks, dilation settings, whether to use extra depth-wise modules, and whether to enable side loss.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/DarkIR.py` |
| Model class name | `DarkIR` |
| Default configuration | `openLLV/deepLearning/config/DarkIR.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/DarkIR_Loss.py` |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "DarkIR",
    "input.jpg",
    output="results/DarkIR/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    model="DarkIR",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="darkir",
    epochs=10,
    batch_size=4,
)
```
