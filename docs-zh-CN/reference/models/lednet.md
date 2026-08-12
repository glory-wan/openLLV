# LEDNet

> Task: 低光图像增强与去模糊（LLIE）

LEDNet 是 openLLV 中结合金字塔池化、曲线注意力与动态滤波的编码器—解码器。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/pdf/2202.03373 |
| 官方源代码 | https://github.com/sczhou/LEDNet |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/LEDNet.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/LEDNet.py` |
| 类名 | `LEDNet` |
| 注册名 | `LEDNet`（无别名；查询忽略大小写及首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/LEDNet_Loss.py` |

## Implementation Notes

三级下采样后接金字塔池化与曲线注意力模块；解码器应用生成的动态核与可选跳跃相加。推理返回张量；训练返回标准输出，仅激活辅助监督时含 `aux.side_output`。前向 `side_loss=None` 仅在配置的 `use_side_loss` 和训练 `mode` 同时为真时启用辅助输出；显式布尔值覆盖这一决定。YAML 将直接构造时的 `use_side_loss=False` 覆盖为 `True`，并省略保留源码默认值的架构键。

## Parameters

| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置映射；非字典抛 `TypeError`；`**kwargs` 优先。 |
| `model_name` | `str` | `"LEDNet"` | 基类元数据。 |
| `input_channels` | `int` | `3` | 输入/输出通道；须为正整数。 |
| `save_dir` | `str` | `"./checkpoints/llie/LEDNet"` | 默认检查点目录。 |
| `channels` | `List[int]` | `[32, 64, 128, 128]` | 四级宽度；须含四个正值。 |
| `connection` | `bool` | `False` | 启用解码器与编码器跳跃相加；要求级间形状/通道兼容，但无显式校验。 |
| `use_side_loss` | `bool` | `False` | 默认辅助监督开关；YAML 覆盖为 `True`。 |
| `mode` | `str` | `"inference"` | 须为 `"train"` 或 `"inference"`。 |
| `kernel_size` | `int` | `5` | 动态卷积核尺寸；须为奇数，源码未显式校验正数。 |
| `curve_n` | `int` | `3` | 曲线注意力迭代次数；须为正。 |
| `ppm_bins` | `Tuple[int, ...]` | `(1, 2, 3, 6)` | 金字塔池化的自适应输出尺寸；无显式校验。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("LEDNet", "input.jpg", output="results/lednet/output.png", config={"connection": True, "curve_n": 4})
```

## Checkpoint / Official Weights

将 openLLV 检查点路径作为预测目标。未实现官方权重自动下载。
