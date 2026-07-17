# DarkIR

> 任务：低光图像增强（`llie`）

DarkIR 是 openLLV 已实现的稳健低光图像恢复模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/pdf/2412.13443 |
| CVPR 论文页面 | https://openaccess.thecvf.com/content/CVPR2025/papers/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.pdf |
| 官方源代码 | https://github.com/cidautai/DarkIR |

## 模型介绍

DarkIR 面向稳健的低光图像恢复。它不仅关注亮度提升，还处理低光场景中常见的噪声、颜色退化和细节丢失。模型采用编码器—解码器结构，并包含多尺度或深度可分离相关模块，以提升暗图像恢复能力。

在 openLLV 中，DarkIR 的默认配置包括网络宽度、编码器与解码器块数量、膨胀设置、是否使用额外的深度方向模块，以及是否启用辅助损失。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/DarkIR.py` |
| 模型类名 | `DarkIR` |
| 默认配置 | `openLLV/deepLearning/config/DarkIR.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/DarkIR_Loss.py` |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "DarkIR",
    "input.jpg",
    output="results/DarkIR/output.png",
    device="cuda",
)
```

训练示例：

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

