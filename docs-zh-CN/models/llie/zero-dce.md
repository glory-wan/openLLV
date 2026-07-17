# Zero-DCE

> 任务：低光图像增强（`llie`）

Zero-DCE 是 openLLV 已实现的零参考低光增强模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf |
| 官方源代码 | https://github.com/Li-Chongyi/Zero-DCE |
| 官方项目页面 | https://li-chongyi.github.io/Proj_Zero-DCE.html |

## 模型介绍

Zero-DCE 将低光增强表述为图像特定的曲线估计问题。该模型不依赖成对的低光/正常光图像进行有监督训练，而是通过一组无参考约束学习逐像素增强曲线。其核心特点是结构轻量、推理快速，并适用于零参考低光增强场景。

在 openLLV 中，Zero-DCE 输出增强图像和曲线参数。训练模式下，模型通过标准输出结构返回 `pred` 以及损失函数所需的中间变量；推理模式下则直接返回增强图像。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/ZeroDCE.py` |
| 模型类名 | `ZeroDCE` |
| 默认配置 | `openLLV/deepLearning/config/ZeroDCE.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/ZeroDCE_Loss.py` |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/ZeroDCE/output.png",
    device="cuda",
)
```

训练示例：

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

