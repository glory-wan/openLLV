# CIDNet

> Task: 低光图像增强（LLIE）

CIDNet 是 openLLV 中在 HVI 空间运行的双分支增强模型。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/abs/2502.20272 |
| 官方源代码 | https://github.com/Fediory/HVI-CIDNet |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/CIDNet.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/CIDNet.py` |
| 类名 | `CIDNet` |
| 注册名 | `CIDNet`（别名：`HVI-CIDNet`；查询忽略大小写及首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/CIDNet_Loss.py` |

## Implementation Notes

模型将 RGB 转为 HVI，在耦合的色相/明度与强度编码器—解码器分支中处理，再转回 RGB。高度和宽度以复制方式填充至 8 的倍数，最终裁剪回原尺寸。`mode="inference"` 时 `forward` 返回张量；`mode="train"` 时返回 `{"pred", "aux", "meta"}`，其中 `aux` 含 `prediction_hvi` 与 `density_k`。仅前向参数 `input_gamma` 可覆盖本次调用的配置值。默认 YAML 与模型默认值一致，但省略 `mode`、`model_name` 和 `save_dir`。

## Parameters

`config` 和 `**kwargs` 覆盖下列默认值，关键字参数优先。未知键会保留在 `model.config`，但若其他 API 不消费则被忽略。

| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置映射；非字典抛 `TypeError`。 |
| `model_name` | `str` | `"CIDNet"` | 存于 `model.config` 的基类元数据。 |
| `input_channels` | `int` | `3` | 输入通道；CIDNet 要求恰为 `3`。 |
| `save_dir` | `str` | `"./checkpoints/llie/CIDNet"` | `LLVModel.save_model` 使用的默认检查点目录。 |
| `channels` | `List[int]` | `[36, 36, 72, 144]` | 四级宽度；它与 `heads` 均须含四个值，转换后的值均须为正。 |
| `heads` | `List[int]` | `[1, 2, 4, 8]` | 四级注意力头数；每个通道数须能被对应头数整除。 |
| `norm` | `bool` | `False` | 经布尔转换后控制上下采样块中的归一化。 |
| `density_k` | `float` | `0.2` | 传给 `HVITransform` 的密度参数；转为 `float`，此处无显式范围校验。 |
| `input_gamma` | `float` | `1.0` | HVI 转换前的伽马；构造时须为正，也可通过 `model_kwargs` 按次覆盖。 |
| `saturation_scale` | `float` | `1.0` | 逆 HVI 转换的饱和度缩放；转为 `float`。 |
| `intensity_scale` | `float` | `1.0` | 逆 HVI 转换的强度缩放；转为 `float`。 |
| `clamp_output` | `bool` | `True` | 为真时将 RGB 结果裁剪至 `[0, 1]`。 |
| `mode` | `str` | `"inference"` | 输出契约；须为 `"train"` 或 `"inference"`。 |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "HVI-CIDNet", "input.jpg", output="results/cidnet/output.png",
    config={"density_k": 0.25, "saturation_scale": 0.9},
    model_kwargs={"input_gamma": 0.9},
)
```

## Checkpoint / Official Weights

可将 openLLV `.pt`/`.pth` 检查点作为 `llv.predict` 的首个参数；显式 `config` 会覆盖其中保存的配置。仓库不会自动下载官方 CIDNet 权重。
