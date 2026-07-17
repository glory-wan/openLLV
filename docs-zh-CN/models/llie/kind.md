# KinD

> 任务：低光图像增强（`llie`）

KinD 是一种受 Retinex 启发、面向实用低光图像增强的深度学习模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1145/3343031.3350926 |
| 论文名称 | Kindling the Darkness: A Practical Low-light Image Enhancer |
| 官方源代码 | https://github.com/zhangyhuaee/KinD |
| 官方项目页面 | 无 |

## 模型介绍

KinD 将图像分解为反射率和照明，恢复退化的反射率，调整照明，然后重新组合两个分量生成增强结果。原始实现分别分阶段训练分解、恢复和照明调整网络。

在 openLLV 中，KinD 被实现为一个集成的 PyTorch 模型，以适配统一训练和预测流水线。模型包含：

| 子网络 | 用途 |
| --- | --- |
| `KinDDecompositionNet` | 估计反射率 `R` 和照明 `I` |
| `KinDRestorationNet` | 在照明引导下恢复低光反射率 |
| `KinDIlluminationAdjustmentNet` | 使用曝光比图调整照明 |

相关损失函数结合官方分阶段训练过程中的主要目标，包括 Retinex 重建、反射率一致性、照明平滑、反射率恢复、照明调整以及最终增强图像监督。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/KinD.py` |
| 模型类名 | `KinD` |
| 默认配置 | `openLLV/deepLearning/config/KinD.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/KinD_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `decomposition_channels` | `int` | `64` | 分解网络使用的特征通道数 |
| `decomposition_layers` | `int` | `5` | 中间分解层数量 |
| `restoration_channels` | `int` | `32` | 恢复网络使用的基础特征通道数 |
| `adjustment_channels` | `int` | `32` | 照明调整网络使用的特征通道数 |
| `adjustment_layers` | `int` | `3` | 中间照明调整层数量 |
| `illumination_ratio` | `float` | `5.0` | 推理时照明调整使用的曝光比 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "KinD",
    "input.jpg",
    output="results/KinD/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    "openLLV/deepLearning/config/KinD.yaml",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=2,
)
```

覆盖推理曝光比：

```python
enhanced, saved_path = llv.predict(
    "KinD",
    "input.jpg",
    output="results/KinD/brighter.png",
    device="cuda",
    illumination_ratio=6.0,
)
```

