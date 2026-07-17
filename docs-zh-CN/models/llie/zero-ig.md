# Zero-IG

> 任务：低光图像增强（`llie`）

Zero-IG 是 openLLV 已实现的零样本低光增强与去噪模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf |
| 官方源代码 | https://github.com/Doyle59217/ZeroIG |

## 模型介绍

Zero-IG 的全称是 Zero-shot Illumination-Guided joint denoising and adaptive enhancement。该方法面向零样本场景，同时考虑低光增强和噪声抑制，并使用照明引导机制指导增强与去噪过程。

在 openLLV 中，Zero-IG 包含增强网络、两阶段去噪网络、纹理差异模块和局部平均池化组件。配置可以设置增强层数、增强通道数、去噪通道数和运行模式。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/ZeroIG.py` |
| 模型类名 | `ZeroIG` |
| 默认配置 | `openLLV/deepLearning/config/ZeroIG.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/ZeroIG_Loss.py` |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroIG",
    "input.jpg",
    output="results/ZeroIG/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    model="ZeroIG",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="zeroig",
    epochs=10,
    batch_size=4,
)
```

