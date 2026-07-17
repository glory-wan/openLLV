# SCI

> Task: low-light image enhancement (`llie`)

SCI is a fast low-light enhancement model already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| Official source code | https://github.com/vis-opt-group/SCI |

## Model Introduction

SCI usually stands for Self-Calibrated Illumination. This model targets fast, flexible, and robust low-light image enhancement, progressively optimizing illumination estimation through an enhancement network and a calibration network. Compared with complex multi-stage models, SCI emphasizes a lightweight structure and efficient inference.

In openLLV, SCI consists of illumination enhancement and calibration parts. The model configuration can control the number of stages, the number of enhancement network layers, the number of calibration network layers, and the channel count.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/SCI.py` |
| Model class name | `SCI` |
| Default configuration | `openLLV/deepLearning/config/SCI.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/Sci_Loss.py` |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "SCI",
    "input.jpg",
    output="results/SCI/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    model="SCI",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="sci",
    epochs=10,
    batch_size=4,
)
```
