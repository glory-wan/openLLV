# LLNet

> 任务：低光图像增强（`llie`）

LLNet 是一种用于自然低光图像增强的堆叠稀疏去噪自编码器模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1016/j.patcog.2016.06.008 |
| 论文 PDF | https://web.me.iastate.edu/soumiks/pdf/Journal/LAS16_llnet.pdf |
| 官方源代码 | https://github.com/kglore/llnet_color |
| 官方项目页面 | 无 |

## 模型介绍

LLNet 使用堆叠稀疏去噪自编码器增强低光图像。原始方法对局部图像块进行扰动和向量化，训练自编码器重建干净的正常光图块，然后根据重叠图块的预测重建完整增强图像。

在 openLLV 中，LLNet 保留了这种逐图块自编码器思想，但将图块提取和聚合封装在 PyTorch 图像到图像模型内部。该实现使用 `torch.nn.functional.unfold` 提取重叠图块，通过全连接编码器—解码器网络处理每个图块，再把重建图块折叠回图像。

相关 openLLV 损失函数使用有监督重建误差和可选的稀疏自编码器正则项。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/LLNet.py` |
| 模型类名 | `LLNet` |
| 默认配置 | `openLLV/deepLearning/config/LLNet.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/LLNet_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `patch_size` | `int` | `17` | 自编码器使用的局部图块大小 |
| `patch_stride` | `int` | `3` | 提取重叠图块时使用的步长 |
| `hidden_dims` | `list[int]` | `[2000, 1600, 1200]` | 编码器隐藏层维度 |
| `activation` | `str` | `"sigmoid"` | 隐藏层激活函数：`sigmoid`、`relu` 或 `tanh` |
| `output_activation` | `str` | `"sigmoid"` | 输出激活函数：`sigmoid`、`clamp` 或 `none` |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LLNet",
    "input.jpg",
    output="results/LLNet/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    "openLLV/deepLearning/config/LLNet.yaml",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=1,
)
```

快速调试时使用更小的隐藏层维度：

```python
llv.train(
    model="LLNet",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="llnet",
    model_params={
        "hidden_dims": [256, 128, 64],
        "patch_stride": 8,
    },
    epochs=2,
    batch_size=1,
)
```

