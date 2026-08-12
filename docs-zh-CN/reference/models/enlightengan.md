# EnlightenGAN

> Task: 低光图像增强（LLIE）

EnlightenGAN 是 openLLV 中带全局与局部判别器的注意力引导生成模型。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TIP.2021.3051462 |
| 官方源代码 | https://github.com/VITA-Group/EnlightenGAN |
| 官方项目页 | https://github.com/VITA-Group/EnlightenGAN |
| 默认配置 | `openLLV/deepLearning/config/EnlightenGAN.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/EnlightenGAN.py` |
| 类名 | `EnlightenGAN` |
| 注册名 | `EnlightenGAN`（声明的别名与其相同；查询忽略大小写） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/EnlightenGAN_Loss.py` |

## Implementation Notes

注意力图为 `(1 - 输入通道均值).clamp(0, 1)`。推理模式只返回增强张量；训练模式返回标准字典，暴露增强图、注意力图、两个判别器模块及确定性的中心局部裁剪。在输入尺寸允许时，裁剪边长至少为 16 像素。默认 YAML 与模型默认值一致，但省略 `mode`、`model_name`、`save_dir`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置映射；非字典抛 `TypeError`；`**kwargs` 优先。 |
| `model_name` | `str` | `"EnlightenGAN"` | 基类元数据。 |
| `input_channels` | `int` | `3` | 输入/输出通道；须为正整数。 |
| `save_dir` | `str` | `"./checkpoints/llie/EnlightenGAN"` | 默认检查点目录。 |
| `generator_channels` | `int` | `32` | 生成器基础宽度；转为 `int` 且须为正。 |
| `discriminator_channels` | `int` | `32` | 两个判别器的基础宽度；转为 `int` 且须为正。 |
| `discriminator_layers` | `int` | `3` | 两个判别器的层数；转为 `int` 且须为正。 |
| `use_attention` | `bool` | `True` | 经布尔转换后控制生成器注意力。 |
| `local_patch_ratio` | `float` | `0.5` | 局部对抗训练的中心裁剪比例；须位于 `(0, 1]`。 |
| `mode` | `str` | `"inference"` | 须为 `"train"` 或 `"inference"`。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("EnlightenGAN", "input.jpg", output="results/enlightengan/output.png", config={"local_patch_ratio": 0.4})
```

## Checkpoint / Official Weights

将 openLLV 检查点路径作为预测目标。未实现官方权重自动下载。
