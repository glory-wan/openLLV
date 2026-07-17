# LEDNet

> 任务：低光图像增强（`llie`）

LEDNet 是 openLLV 已实现的低光增强与去模糊模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/pdf/2202.03373 |
| 官方源代码 | https://github.com/sczhou/LEDNet |

## 模型介绍

LEDNet 面向低光图像增强和去模糊任务。其设计包含多尺度特征提取、动态卷积或注意力相关模块，以同时改善暗场景中的亮度、细节和清晰度。

在 openLLV 中，LEDNet 的配置包括通道设置、跳跃连接、辅助损失、动态卷积核大小、曲线注意力迭代次数以及金字塔池化分箱设置。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/LEDNet.py` |
| 模型类名 | `LEDNet` |
| 默认配置 | `openLLV/deepLearning/config/LEDNet.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/LEDNet_Loss.py` |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LEDNet",
    "input.jpg",
    output="results/LEDNet/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    model="LEDNet",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="lednet",
    epochs=10,
    batch_size=4,
)
```

