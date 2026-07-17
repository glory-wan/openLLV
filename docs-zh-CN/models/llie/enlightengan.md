# EnlightenGAN

> 任务：低光图像增强（`llie`）

EnlightenGAN 是一种面向非成对低光图像增强的注意力引导生成对抗模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TIP.2021.3051462 |
| 论文名称 | EnlightenGAN: Deep Light Enhancement Without Paired Supervision |
| 官方源代码 | https://github.com/VITA-Group/EnlightenGAN |
| 官方项目页面 | https://github.com/VITA-Group/EnlightenGAN |

## 模型介绍

EnlightenGAN 无须成对的低光和正常光监督即可增强低光图像。原始方法使用注意力引导的 U-Net 生成器、全局和局部判别器，以及自特征保持正则化，使增强图像在变亮的同时保留输入图像的结构和颜色特征。

在 openLLV 中，EnlightenGAN 已适配统一的模型和训练器接口。模型包含：

| 组件 | 用途 |
| --- | --- |
| `EnlightenGANGenerator` | 注意力引导的 U-Net 生成器 |
| `global_discriminator` | 面向完整图像的 PatchGAN 判别器 |
| `local_discriminator` | 面向裁剪后局部图块的 PatchGAN 判别器 |
| 注意力图 | 根据输入图像估计的暗度感知图 |

相关损失函数结合全局/局部 LSGAN 损失、判别器损失、自特征保持正则化、曝光正则化和总变差平滑项。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/EnlightenGAN.py` |
| 模型类名 | `EnlightenGAN` |
| 默认配置 | `openLLV/deepLearning/config/EnlightenGAN.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/EnlightenGAN_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `generator_channels` | `int` | `32` | 生成器使用的基础特征通道数 |
| `discriminator_channels` | `int` | `32` | 判别器使用的基础特征通道数 |
| `discriminator_layers` | `int` | `3` | PatchGAN 下采样层数 |
| `use_attention` | `bool` | `True` | 是否将注意力图拼接到生成器输入 |
| `local_patch_ratio` | `float` | `0.5` | 局部对抗损失使用的中心局部图块比例 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "EnlightenGAN",
    "input.jpg",
    output="results/EnlightenGAN/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    "openLLV/deepLearning/config/EnlightenGAN.yaml",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=2,
)
```

快速调试配置：

```python
llv.train(
    model="EnlightenGAN",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="enlightengan",
    model_params={
        "generator_channels": 16,
        "discriminator_channels": 16,
    },
    epochs=2,
    batch_size=1,
)
```

