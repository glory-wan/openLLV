# LLFlow

> 任务：低光图像增强（`llie`）

LLFlow 是一种用于低光图像增强的条件归一化流模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1609/aaai.v36i3.20162 |
| 论文名称 | Low-Light Image Enhancement with Normalizing Flow |
| 官方源代码 | https://github.com/wyf0912/LLFlow |
| 官方项目页面 | https://github.com/wyf0912/LLFlow |

## 模型介绍

LLFlow 使用归一化流，对给定低光图像条件下正常曝光图像的条件分布进行建模。它不仅预测单个确定性输出，还在低光输入的条件下学习图像空间与潜在空间之间的可逆映射。原始方法使用负对数似然目标训练流模型，并可以通过采样或使用确定性潜在编码生成增强图像。

在 openLLV 中，LLFlow 已适配统一的图像到图像模型接口。实现包含：

| 组件 | 用途 |
| --- | --- |
| `LLFlowConditionEncoder` | 提取低光条件特征 |
| `LLFlowAffineCoupling` | 条件仿射耦合流层 |
| `LLFlow` | 堆叠条件流层并执行正向/反向变换 |

训练期间，损失函数将成对的正常光目标映射到潜在空间，并计算流模型的负对数似然。推理期间，模型默认使用全零潜在张量生成确定性增强结果。设置 `sample_temperature > 0` 可以采样随机输出。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/LLFlow.py` |
| 模型类名 | `LLFlow` |
| 默认配置 | `openLLV/deepLearning/config/LLFlow.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/LLFlow_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `condition_channels` | `int` | `32` | 低光条件特征通道数 |
| `condition_blocks` | `int` | `4` | 条件编码器中的残差块数量 |
| `flow_layers` | `int` | `8` | 条件仿射耦合层数量 |
| `flow_hidden_channels` | `int` | `64` | 每个耦合网络中的隐藏通道数 |
| `scale_clamp` | `float` | `2.0` | 仿射耦合对数尺度的裁剪范围 |
| `sample_temperature` | `float` | `0.0` | 推理期间使用的潜在采样温度 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LLFlow",
    "input.jpg",
    output="results/LLFlow/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    "openLLV/deepLearning/config/LLFlow.yaml",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=2,
)
```

随机推理：

```python
enhanced, saved_path = llv.predict(
    "LLFlow",
    "input.jpg",
    output="results/LLFlow/sample.png",
    device="cuda",
    sample_temperature=0.7,
)
```

