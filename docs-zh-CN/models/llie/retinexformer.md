# RetinexFormer

> 任务：低光图像增强（`llie`）

RetinexFormer 是一种用于低光图像增强的单阶段 Retinex Transformer 模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.pdf |
| 官方源代码 | https://github.com/caiyuanhao1998/Retinexformer |
| 官方项目页面 | 无 |

## 模型介绍

RetinexFormer 将基于 Retinex 的照明建模与 Transformer 风格的特征恢复相结合。模型首先估计照明特征和照明图，然后使用照明引导的多头自注意力指导图像恢复。与多阶段 Retinex 分解流水线相比，RetinexFormer 被设计为用于低光图像增强的单阶段架构。

在 openLLV 中，RetinexFormer 被实现为已注册的深度学习模型。该实现包含照明估计器、照明引导注意力块、U-Net 风格去噪器以及可选的多阶段堆叠。训练模式下，模型通过标准 openLLV 模型输出字典返回增强图像和各阶段中间输出；推理模式下返回增强图像张量。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/RetinexFormer.py` |
| 模型类名 | `RetinexFormer` |
| 默认配置 | `openLLV/deepLearning/config/RetinexFormer.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/RetinexFormer_Loss.py` |

## 损失函数

官方 RetinexFormer 训练配置使用像素级 L1 损失。openLLV 将该损失注册为 `retinexformer`，并通过 `illumination_weight` 提供可选的照明一致性项。

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "RetinexFormer",
    "input.jpg",
    output="results/RetinexFormer/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    model="RetinexFormer",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="retinexformer",
    epochs=10,
    batch_size=4,
)
```

YAML 配置：

```python
llv.train("openLLV/deepLearning/config/RetinexFormer.yaml")
```

