# SCI

> 任务：低光图像增强（LLIE）

SCI 是分阶段进行照明估计与校准的模型，在 openLLV 中以注册名 `SCI` 选择。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| 官方源码 | https://github.com/vis-opt-group/SCI |
| 官方项目页 | 无 |
| 默认配置 | `openLLV/deepLearning/config/SCI.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/SCI.py` |
| 类名 | `SCI` |
| 注册名 | `SCI`（无别名；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/Sci_Loss.py` 中的 `Sci_Loss`（注册名 `sci`；别名 `sci_loss`） |

## Implementation Notes

构造器为 `SCI(config=None, **kwargs)`。`LLVModel` 依次合并共享默认值、`config` 和 `kwargs`，因此关键字覆盖值优先于 `config` 中的同名键。增强器估计照明图，训练时每个阶段再校准下一阶段输入。`mode="train"` 时，`forward(x)` 返回标准字典：`pred` 为最终反射图，`aux` 含 `enhanced`、`ilist`、`rlist`、`inlist` 和 `attlist`。`mode="inference"` 时只返回第一阶段增强张量。网络层硬编码为三通道输入；更改 `input_channels` 不会改变这些层。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 覆盖全部默认值的配置字典。 | 非字典且非 `None` 会抛 `TypeError`；未知键会保留，但未被模型代码消费时不起作用。 |
| `**kwargs` | `Any` | `{}` | 在 `config` 之后合并的扁平配置覆盖值。 | 键语义与 `config` 相同。 |
| `model_name` | `str` | `"SCI"` | 存于 `model.config` 的 `LLVModel` 共享元数据。 | 不校验，也不参与架构构造。 |
| `input_channels` | `int` | `3` | 共享输入通道元数据。 | 必须为正整数，否则抛 `ValueError`；SCI 层仍要求三通道张量。 |
| `save_dir` | `str` | `"./checkpoints/llie/SCI"` | `save_model()` 与 `save_config()` 的默认目录。 | 构造时不校验。 |
| `stage` | `int` | `3` | 训练时的分阶段增强/校准迭代数。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `enhance_layers` | `int` | `1` | 增强网络残差块数量。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `enhance_channels` | `int` | `3` | 增强网络特征宽度。 | 无显式校验；必须能被 PyTorch 卷积与批归一化构造器接受。 |
| `calibrate_layers` | `int` | `3` | 校准网络残差块数量。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `calibrate_channels` | `int` | `16` | 校准网络特征宽度。 | 无显式校验；必须能被 PyTorch 层构造器接受。 |
| `mode` | `str` | `"inference"` | 选择前向输出契约。 | 只能是 `"train"` 或 `"inference"`，否则抛 `ValueError`。 |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("SCI", "input.jpg", output="results/sci/output.png", config={"stage": 2, "mode": "inference"})
```

```python
import openLLV as llv

result = llv.train("SCI", model_params={"stage": 3, "mode": "train"})
```

## Checkpoint / Official Weights

可将 openLLV `.pt` 或 `.pth` 检查点作为预测方法传入。加载时恢复保存的模型类与配置；显式 predictor `config` 会覆盖已保存配置。本实现不会自动下载 SCI 官方权重。
