# URetinex-Net

> Task: low-light image enhancement (`llie`)

URetinex-Net is a Retinex-based deep unfolding low-light enhancement model already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| Official source code | https://github.com/AndersonYong/URetinex-Net |

## Model Introduction

URetinex-Net is based on Retinex theory and formulates low-light image enhancement as an unfoldable optimization process. Through multiple unfolding rounds, the model jointly models reflectance, illumination, and enhancement results, making it suitable for low-light enhancement under complex illumination degradation.

In openLLV, the implementation class name of URetinex-Net is `URetinexNet`. Its configuration can set the number of unfolding rounds, Retinex-related weights, illumination adjustment ratio, and whether to use adaptive ratio.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/URetinex.py` |
| Model class name | `URetinexNet` |
| Default configuration | `openLLV/deepLearning/config/URetinexNet.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/URetinex_Loss.py` |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "URetinexNet",
    "input.jpg",
    output="results/URetinexNet/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    model="URetinexNet",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="uretinex",
    epochs=10,
    batch_size=4,
)
```
