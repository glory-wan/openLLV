# Zero-DCE

> Task: low-light image enhancement (`llie`)

Zero-DCE is a zero-reference low-light enhancement model already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf |
| Official source code | https://github.com/Li-Chongyi/Zero-DCE |
| Official project page | https://li-chongyi.github.io/Proj_Zero-DCE.html |

## Model Introduction

Zero-DCE formulates low-light enhancement as an image-specific curve estimation problem. The model does not rely on paired low-light/normal-light images for supervised training; instead, it learns pixel-wise enhancement curves through a set of no-reference constraints. Its core characteristics are a lightweight structure, fast inference, and suitability for zero-reference low-light enhancement scenarios.

In openLLV, Zero-DCE outputs the enhanced image and curve parameters. In training mode, the model returns `pred` and intermediate variables needed by the loss through the standard output structure; in inference mode, it directly returns the enhanced image.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/ZeroDCE.py` |
| Model class name | `ZeroDCE` |
| Default configuration | `openLLV/deepLearning/config/ZeroDCE.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/ZeroDCE_Loss.py` |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/ZeroDCE/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    model="ZeroDCE",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="zerodce",
    epochs=10,
    batch_size=4,
)
```
