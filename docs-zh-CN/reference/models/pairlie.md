# PairLIE

> 任务：低光图像增强（LLIE）

PairLIE 是从成对低光实例训练、可用于单图推理的 Retinex 风格分解模型。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf |
| 官方源码 | https://github.com/zhenqifu/PairLIE |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/PairLIE.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/PairLIE.py` |
| 类名 | `PairLIE` |
| 注册名 | `PairLIE`（别名：`Pair-LIE`） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/PairLIE_Loss.py`（`pairlie`；别名：`pairlie_loss`、`PairLIE-Loss`） |

## Implementation Notes

三个五层卷积估计器分别预测去噪图、单通道照明图和三通道反射图。增强计算为 `illumination ** enhancement_gamma * reflectance`。推理模式下，`forward(image)` 只返回预测。训练模式返回 `{"pred", "aux", "meta"}`；提供 `paired_image` 后，`aux` 还包含第二张图的分解结果。`PairLIE_Loss` 需要该成对分解，因此统一训练必须提供第二个低光实例。类属性 `requires_paired_forward=True` 声明了这一点。

估计器模块名与官方实现一致，以兼容原始状态字典。构造器为 `PairLIE(config=None, **kwargs)`，关键字值覆盖 `config`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置字典。 | 必须是字典或 `None`。 |
| `model_name` | `str` | `"PairLIE"` | 继承自 `LLVModel` 的模型名。 | 无额外校验。 |
| `input_channels` | `int` | `3` | 输入通道数。 | 必须恰为 `3`，否则抛出 `ValueError`；内部估计器固定为 RGB。 |
| `save_dir` | `str` | `"./checkpoints/llie/PairLIE"` | 默认检查点/配置目录。 | 无额外校验。 |
| `feature_channels` | `int` | `64` | 三个估计器共享的隐藏宽度。 | 经 `int()` 转换后必须为正数。 |
| `enhancement_gamma` | `float` | `0.2` | 合成时施加于照明图的指数。 | 经 `float()` 转换后必须为正数。 |
| `clamp_output` | `bool` | `True` | 将合成预测截断至 `[0, 1]`。 | 按真值判断，无严格类型检查。 |
| `mode` | `str` | `"inference"` | 选择单一预测或结构化训练输出。 | 只能是 `"train"` 或 `"inference"`。 |
| `forward(..., paired_image=...)` | `Optional[torch.Tensor]` | `None` | 训练分解所需的同场景第二个低光实例。 | 模型层面可选；缺少成对目标时 `PairLIE_Loss` 抛 `ValueError`，缺少成对分解字段时抛 `KeyError`。 |

所有配置键均可经构造器 `**kwargs` 传入；未知键会保留但不使用。其他前向 `**kwargs` 会被接受并忽略。

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "Pair-LIE",
    "input.jpg",
    output="results/pairlie/output.png",
    config={"enhancement_gamma": 0.25, "clamp_output": True},
)
```

```python
from openLLV.deepLearning.models.LLIE.PairLIE import PairLIE

model = PairLIE(mode="train")
training_output = model(first_low_light_tensor, paired_image=second_low_light_tensor)
```

## Checkpoint / Official Weights

可将 openLLV 检查点直接传给 `llv.predict`。虽然估计器名称保留官方原始状态字典命名，openLLV 不会自动下载或加载原始官方检查点。
