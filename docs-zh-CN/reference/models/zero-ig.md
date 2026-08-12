# Zero-IG

> 任务：低光图像增强（LLIE）

Zero-IG 将照明引导增强与两个去噪网络结合，在 openLLV 中注册为 `ZeroIG`。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf |
| 官方源码 | https://github.com/Doyle59217/ZeroIG |
| 官方项目页 | 无 |
| 默认配置 | `openLLV/deepLearning/config/ZeroIG.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/ZeroIG.py` |
| 类名 | `ZeroIG` |
| 注册名 | `ZeroIG`（无别名；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/ZeroIG_Loss.py` 中的 `ZeroIG_Loss`（注册名 `zeroig`；别名 `zeroig_loss`、`zero_ig`、`zero-ig`、`ZeroIG_Loss`） |

## Implementation Notes

`ZeroIG(config=None, **kwargs)` 在 `config` 之后合并关键字覆盖值。训练模式会初始化权重并返回标准字典，其中 `aux.outputs` 是 `ZeroIG_Loss` 消费的 21 张量元组；推理返回最终三通道增强/去噪张量；微调返回含 `H2` 和 `H3` 的标准字典。微调模式下，真值 `pretrained_weights` 路径会在构造时立即由 `torch.load` 加载；仅复制当前状态字典存在的键，加载失败只打印而不继续抛出。`set_mode()` 可在构造后改变前向路由，但不会初始化权重或加载微调权重。尽管存在共享 `input_channels` 元数据，各子网络均按三通道 RGB 硬编码。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 覆盖默认值的字典。 | 非字典且非 `None` 抛 `TypeError`；未知键保留但未被消费时不起作用。 |
| `**kwargs` | `Any` | `{}` | 在 `config` 后合并的覆盖值。 | 键语义与 `config` 相同。 |
| `model_name` | `str` | `"ZeroIG"` | 共享模型元数据。 | 不校验，也不参与架构构造。 |
| `input_channels` | `int` | `3` | 共享输入通道元数据。 | 必须为正整数；模型层仍假定三通道输入。 |
| `save_dir` | `str` | `"./checkpoints/llie/ZeroIG"` | 默认检查点/配置输出目录。 | 构造时不校验。 |
| `enhance_layers` | `int` | `3` | 残差增强块数量。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `enhance_channels` | `int` | `64` | 增强网络特征宽度。 | 无显式校验；必须能被 PyTorch 层构造器接受。 |
| `denoise1_channels` | `int` | `48` | 第一去噪器特征宽度。 | 无显式校验；必须能被 PyTorch 层构造器接受。 |
| `denoise2_channels` | `int` | `48` | 第二去噪器特征宽度。 | 无显式校验；必须能被 PyTorch 层构造器接受。 |
| `mode` | `str` | `"inference"` | 选择训练、推理或微调行为及输出。 | 只能是 `"train"`、`"inference"` 或 `"finetune"`，否则抛 `ValueError`。 |
| `pretrained_weights` | `Optional[str]` | `None` | 微调模式构造期间加载的状态字典路径。 | 仅当 `mode="finetune"` 且值为真时加载；错误会打印并被抑制。 |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("ZeroIG", "input.jpg", output="results/zero-ig/output.png", config={"enhance_layers": 4, "mode": "inference"})
```

```python
import openLLV as llv

result = llv.train("ZeroIG", model_params={"mode": "train"})
```

## Checkpoint / Official Weights

普通 openLLV 检查点可作为 `.pt` 或 `.pth` 路径传给 `llv.predict`。`pretrained_weights` 是另一条微调加载路径，它需要从状态字典键直接映射到张量的原始字典，而不是 openLLV 检查点包装。未实现自动下载。
