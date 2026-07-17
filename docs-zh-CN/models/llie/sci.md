# SCI

> 任务：低光图像增强（`llie`）

SCI 是 openLLV 已实现的快速低光增强模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| 官方源代码 | https://github.com/vis-opt-group/SCI |

## 模型介绍

SCI 通常指 Self-Calibrated Illumination（自校准照明）。该模型面向快速、灵活且稳健的低光图像增强，通过增强网络和校准网络逐步优化照明估计。与复杂的多阶段模型相比，SCI 更强调轻量结构和高效推理。

在 openLLV 中，SCI 由照明增强和校准两部分组成。模型配置可以控制阶段数、增强网络层数、校准网络层数和通道数量。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/SCI.py` |
| 模型类名 | `SCI` |
| 默认配置 | `openLLV/deepLearning/config/SCI.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/Sci_Loss.py` |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "SCI",
    "input.jpg",
    output="results/SCI/output.png",
    device="cuda",
)
```

训练示例：

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

