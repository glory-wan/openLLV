# LLFormer

> 任务：低光图像增强（LLIE）

LLFormer 是支持整图或分块低光增强的轴向注意力 Transformer 模型。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/abs/2212.11548 |
| 官方源码 | https://github.com/TaoWangzj/LLFormer |
| 官方项目页 | https://taowangzj.github.io/projects/LLFormer/ |
| 默认配置 | `openLLV/deepLearning/config/LLFormer.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/LLFormer.py` |
| 类名 | `LLFormer` |
| 注册名 | `LLFormer`（别名：`LL-Former`） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/LLFormer_Loss.py`（`llformer`；别名：`llformer_loss`、`LLFormer-Loss`） |

## Implementation Notes

LLFormer 使用四个分辨率层级、轴向注意力、双门控前馈块、跨层融合和可学习加权跳连。`pad_input=True` 时，输入补齐至 16 的倍数，输出再裁回原尺寸。仅在推理模式且 `tile_size` 非 `None` 时启用分块推理；重叠预测取平均。训练模式返回标准 `{"pred", "aux", "meta"}` 字典且不截断预测值。推理返回张量，并可截断至 `[0, 1]`。

构造器为 `LLFormer(config=None, **kwargs)`，依次合并默认值、`config` 和 `**kwargs`。注册表查询忽略大小写并去除首尾空白。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置字典。 | 必须是字典或 `None`。 |
| `model_name` | `str` | `"LLFormer"` | `LLVModel` 保存的模型名。 | 无额外校验。 |
| `input_channels` | `int` | `3` | 输入通道数。 | 基类要求为正整数。 |
| `save_dir` | `str` | `"./checkpoints/llie/LLFormer"` | 默认检查点/配置目录。 | 无额外校验。 |
| `output_channels` | `int` | `3` | 输出通道数。 | 经 `int()` 转换后必须为正数。 |
| `dim` | `int` | `16` | 基础特征宽度。 | 经 `int()` 转换后必须为正偶数。 |
| `num_blocks` | `List[int]` | `[2, 4, 8, 16]` | 四层 Transformer 块数量。 | 必须可迭代，恰含四个可转为正整数的值。 |
| `num_refinement_blocks` | `int` | `2` | 每个细化分支的块数。 | 经 `int()` 转换后必须为正数。 |
| `heads` | `List[int]` | `[1, 2, 4, 8]` | 四层注意力头数。 | 恰含四个正整数；每个值须整除对应宽度 `[dim, 2*dim, 4*dim, 8*dim]`。 |
| `ffn_expansion_factor` | `float` | `2.66` | 前馈隐藏宽度乘数。 | 经 `float()` 转换后必须为正数。 |
| `bias` | `bool` | `False` | 为前馈及部分卷积层启用偏置；轴向注意力始终使用偏置。 | 构造时经 `bool()` 转换。 |
| `layernorm_type` | `str` | `"WithBias"` | 层归一化实现。 | 只能是 `"BiasFree"` 或 `"WithBias"`。 |
| `attention` | `bool` | `True` | 控制融合系数是否需要梯度。 | 经 `bool()` 转换；不会关闭轴向注意力模块。 |
| `skip` | `bool` | `False` | 将输入图像加到核心输出上。 | 为真时输入与输出通道值必须相等。 |
| `pad_input` | `bool` | `True` | 前向前将空间尺寸补齐到 16 的倍数。 | 按真值判断；关闭时输入尺寸须与架构兼容。 |
| `clamp_output` | `bool` | `True` | 将推理输出截断至 `[0, 1]`。 | 仅用于推理模式。 |
| `tile_size` | `Optional[Union[int, Tuple[int, int]]]` | `None` | 默认推理分块高宽；整数表示正方形。 | `None`、正整数或两个可转为正整数的值；尺寸须被 16 整除。 |
| `tile_overlap` | `Union[int, Tuple[int, int]]` | `0` | 默认分块重叠；整数用于两个轴。 | 一个或两个非负整数；构造时配置分块后，每个值须小于相应分块尺寸。 |
| `mode` | `str` | `"inference"` | 选择结构化训练输出或张量推理输出。 | 只能是 `"train"` 或 `"inference"`。 |
| `forward(..., tile_size=...)` | `Optional[Union[int, Tuple[int, int]]]` | 当前 `config["tile_size"]` | 单次前向的分块覆盖值，可经 `llv.predict` 的 `model_kwargs` 传入。 | 与 `tile_size` 相同；仅推理模式使用。 |
| `forward(..., tile_overlap=...)` | `Union[int, Tuple[int, int]]` | 当前 `config["tile_overlap"]` | 单次前向的重叠覆盖值。 | 与 `tile_overlap` 相同；运行时代码不会重复构造阶段的“小于分块尺寸”校验。 |

经构造器 `**kwargs` 提供的配置键覆盖 `config` 同名键。其他未知键会保留但不使用。

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LL-Former",
    "input.jpg",
    output="results/llformer/output.png",
    config={"tile_size": 512, "tile_overlap": 64},
)
```

```python
import openLLV as llv

enhanced, _ = llv.predict(
    "LLFormer",
    "input.jpg",
    save=False,
    model_kwargs={"tile_size": (512, 768), "tile_overlap": (64, 64)},
)
```

## Checkpoint / Official Weights

openLLV 检查点可直接传给 `llv.predict`。本实现保留模型架构，但不会自动下载或重映射官方权重。
