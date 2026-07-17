# KinD++

> 任务：低光图像增强（`llie`）

KinD++ 是一种改进的 Retinex 低光增强模型，它通过多尺度照明注意力扩展了 KinD。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1007/s11263-020-01407-x |
| 论文名称 | Beyond Brightening Low-light Images |
| 官方源代码 | https://github.com/zhangyhuaee/KinD_plus |
| 官方项目页面 | 无 |

## 模型介绍

KinD++ 改进了原始 KinD 框架，使低光增强更加稳健。它保留 Retinex 风格的分解—恢复—调整流水线，但在反射率恢复网络中引入多尺度照明注意力模块（MSIA）。MSIA 使用估计的照明图引导多尺度特征恢复，从而减少非均匀斑点和过度平滑伪影。

在 openLLV 中，由于 Python 类名不能包含 `++`，该模型实现为 `KinDPlusPlus`。默认 YAML 文件保留论文风格名称：

```text
openLLV/deepLearning/config/KinD++.yaml
```

openLLV 实现包含：

| 子网络 | 用途 |
| --- | --- |
| `KinDPPDecompositionNet` | 估计反射率和照明 |
| `KinDPPRestorationNet` | 使用 MSIA 引导特征恢复反射率 |
| `KinDPPIlluminationAdjustmentNet` | 使用曝光比图调整照明 |

相关损失函数结合官方分阶段训练代码使用的主要目标：分解重建、反射率一致性、相互照明约束、输入感知的照明平滑、反射率恢复、照明调整以及最终增强图像监督。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/KinDPlusPlus.py` |
| 模型类名 | `KinDPlusPlus` |
| 默认配置 | `openLLV/deepLearning/config/KinD++.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/KinDPlusPlus_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `decomposition_channels` | `int` | `32` | 分解网络使用的基础特征通道数 |
| `restoration_channels` | `int` | `32` | MSIA 恢复网络使用的基础特征通道数 |
| `adjustment_channels` | `int` | `32` | 照明调整网络使用的特征通道数 |
| `illumination_ratio` | `float` | `5.0` | 推理时照明调整使用的曝光比 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "KinDPlusPlus",
    "input.jpg",
    output="results/KinDPlusPlus/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    "openLLV/deepLearning/config/KinD++.yaml",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=2,
)
```

覆盖推理曝光比：

```python
enhanced, saved_path = llv.predict(
    "KinDPlusPlus",
    "input.jpg",
    output="results/KinDPlusPlus/brighter.png",
    device="cuda",
    illumination_ratio=6.0,
)
```

