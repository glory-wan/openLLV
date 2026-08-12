# LLFlow

> 任务：低光图像增强（LLIE）

LLFlow 是在 openLLV 中注册的条件归一化流低光图像增强模型。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1609/aaai.v36i3.20162 |
| 官方源码 | https://github.com/wyf0912/LLFlow |
| 官方项目页 | https://wyf0912.github.io/LLFlow/ |
| 默认配置 | `openLLV/deepLearning/config/LLFlow.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/LLFlow.py` |
| 类名 | `LLFlow` |
| 注册名 | `LLFlow`（无别名） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/LLFlow_Loss.py`（`llflow`；别名：`llflow_loss`、`low_light_flow`、`normalizing_flow_loss`） |

## Implementation Notes

条件编码器提取低光特征，交替的仿射耦合层在图像空间与潜空间之间变换。输入至少需要两个通道。推理模式下，`sample_temperature=0.0` 时潜变量全为零，否则使用按该温度缩放的高斯噪声。训练模式下，`forward()` 返回 `{"pred", "aux", "meta"}`；`aux` 包含条件特征、潜变量及 `LLFlow_Loss` 所需的正向/逆向流可调用对象。推理模式只返回 `[0, 1]` 内的增强张量。

构造器为 `LLFlow(config=None, **kwargs)`。合并顺序是默认配置、`config`、`**kwargs`，因此关键字覆盖优先。`config` 必须是字典或 `None`。未知键会保留在 `model.config` 中，但本模型不会消费它们。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 模型配置字典。 | 非字典且非 `None` 时抛出 `TypeError`。 |
| `model_name` | `str` | `"LLFlow"` | 继承自 `LLVModel` 的模型名称。 | 无额外校验。 |
| `input_channels` | `int` | `3` | 输入和流变换的通道数。 | 必须是大于等于 `2` 的整数，否则抛出 `ValueError`。 |
| `save_dir` | `str` | `"./checkpoints/llie/LLFlow"` | 继承自 `LLVModel` 的默认检查点/配置输出目录。 | 无额外校验。 |
| `condition_channels` | `int` | `32` | 条件编码器特征宽度。 | 经 `int()` 转换后必须为正数。 |
| `condition_blocks` | `int` | `4` | 条件编码器残差块数量。 | 经 `int()` 转换后必须为正数。 |
| `flow_layers` | `int` | `8` | 仿射耦合层数量。 | 经 `int()` 转换后必须为正数。 |
| `flow_hidden_channels` | `int` | `64` | 每个耦合网络的隐藏宽度。 | 经 `int()` 转换后必须为正数。 |
| `scale_clamp` | `float` | `2.0` | 预测对数尺度的 tanh 截断值。 | 经 `float()` 转换后必须为正数。 |
| `sample_temperature` | `float` | `0.0` | 推理潜变量采样的标准差乘数。 | 经 `float()` 转换后必须非负。 |
| `mode` | `str` | `"inference"` | 选择张量推理结果或结构化训练结果。 | 只能是 `"train"` 或 `"inference"`。 |

所有配置键也可经构造器 `**kwargs` 直接提供，并覆盖 `config` 中的同名键。

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LLFlow",
    "input.jpg",
    output="results/llflow/output.png",
    config={"sample_temperature": 0.1},
)
```

```python
from openLLV.deepLearning.models.LLIE.LLFlow import LLFlow

model = LLFlow(condition_channels=48, flow_layers=6, mode="train")
```

## Checkpoint / Official Weights

可将 openLLV `.pt`/`.pth` 检查点作为 `llv.predict` 的第一个参数。检查点保存类名、合并后的配置和状态字典。本实现不会自动下载官方权重。
