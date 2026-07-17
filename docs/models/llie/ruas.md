# RUAS

> Task: low-light image enhancement (`llie`)

RUAS is a Retinex-inspired low-light enhancement model already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf |
| Official source code | https://github.com/KarelZhang/RUAS |

## Model Introduction

RUAS combines Retinex ideas, unrolled optimization, and architecture search for low-light image enhancement. The model includes enhancement- and denoising-related structures, simulates iterative optimization through unrolling, and introduces cooperative prior architecture search to obtain an effective network structure.

In openLLV, RUAS contains illumination enhancement and denoising modules. The default configuration can control the number of IEM iterations, the number of NRM layers, enhancement network channels, and denoising network channels.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/RUAS.py` |
| Model class name | `RUAS` |
| Default configuration | `openLLV/deepLearning/config/RUAS.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/RUAS_Loss.py` |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "RUAS",
    "input.jpg",
    output="results/RUAS/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    model="RUAS",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="ruas",
    epochs=10,
    batch_size=4,
)
```
