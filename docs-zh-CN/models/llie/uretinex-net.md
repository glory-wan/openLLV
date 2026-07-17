# URetinex-Net

> 任务：低光图像增强（`llie`）

URetinex-Net 是 openLLV 已实现的基于 Retinex 的深度展开低光增强模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| 官方源代码 | https://github.com/AndersonYong/URetinex-Net |

## 模型介绍

URetinex-Net 基于 Retinex 理论，将低光图像增强表述为可展开的优化过程。模型通过多轮展开联合建模反射率、照明和增强结果，适用于复杂照明退化条件下的低光增强。

在 openLLV 中，URetinex-Net 的实现类名为 `URetinexNet`。其配置可以设置展开轮数、Retinex 相关权重、照明调整比例，以及是否使用自适应比例。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/URetinex.py` |
| 模型类名 | `URetinexNet` |
| 默认配置 | `openLLV/deepLearning/config/URetinexNet.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/URetinex_Loss.py` |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "URetinexNet",
    "input.jpg",
    output="results/URetinexNet/output.png",
    device="cuda",
)
```

训练示例：

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

