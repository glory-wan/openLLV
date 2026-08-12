# RetinexFormer

> 任务：低光图像增强（LLIE）

RetinexFormer 是将照明估计与照明引导去噪结合的分阶段 Retinex Transformer。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.pdf |
| 官方源码 | https://github.com/caiyuanhao1998/Retinexformer |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/RetinexFormer.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/RetinexFormer.py` |
| 类名 | `RetinexFormer` |
| 注册名 | `RetinexFormer`（无别名） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/RetinexFormer_Loss.py`（`retinexformer`；别名：`retinexformer_loss`、`RetinexFormer_Loss`、`retinexformer_l1`） |

## Implementation Notes

每个阶段估计照明特征和照明图，形成 `image * illumination_map + image`，再应用照明引导的 U-Net 风格去噪器。空间尺寸以反射方式补齐到 `2 ** levels` 的倍数，然后裁回原尺寸。多个阶段依次消费前一阶段输出。训练返回含每阶段照明中间量的 `{"pred", "aux", "meta"}`；推理返回张量。启用输出截断时，两种模式都会在返回前截断。

`num_blocks` 在去噪器内归一化：所有值经整数转换；序列不足 `levels + 1` 时重复末值，过长则截断。空序列会触发 `IndexError`，源码未显式校验正数。构造器依次合并默认值、`config`、`**kwargs`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置字典。 | 必须是字典或 `None`。 |
| `model_name` | `str` | `"RetinexFormer"` | 继承自 `LLVModel` 的模型名。 | 无额外校验。 |
| `input_channels` | `int` | `3` | 输入通道数。 | 基类要求为正整数；阶段间和残差相加实际还要求其与 `output_channels` 匹配。 |
| `save_dir` | `str` | `"./checkpoints/llie/RetinexFormer"` | 默认检查点/配置目录。 | 无额外校验。 |
| `output_channels` | `int` | `3` | 恢复图像通道数。 | 与零比较并要求为正数；不兼容的非数值类型可能抛 `TypeError`。 |
| `feature_channels` | `int` | `32` | 基础特征宽度和注意力头维度。 | 与零比较并要求为正数；构造时经 `int()` 转换。 |
| `stage` | `int` | `1` | 顺序 RetinexFormer 阶段数量。 | 与零比较并要求为正数；构造时经 `int()` 转换。 |
| `levels` | `int` | `2` | 编码器/解码器深度；填充因子为 `2 ** levels`。 | 与零比较并要求为正数；经 `int()` 转换。 |
| `num_blocks` | `Sequence[int]` | `[1, 1, 1]` | 各编码层和瓶颈的注意力块数。 | 必须是非空且各值可转整数的可迭代对象；扩展/截断为 `levels + 1`，未校验正数。 |
| `mode` | `str` | `"inference"` | 选择结构化训练输出或张量推理输出。 | 只能是 `"train"` 或 `"inference"`。 |
| `clamp_output` | `bool` | `True` | 将最终输出截断至 `[0, 1]`。 | 通过 `config.get(..., True)` 读取，两种模式均按真值判断。 |

经构造器 `**kwargs` 传入的配置键覆盖 `config`；未知键会保留但不使用。前向 `**kwargs` 会被接受并忽略。

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "RetinexFormer",
    "input.jpg",
    output="results/retinexformer/output.png",
    config={"stage": 2, "num_blocks": [1, 2, 2]},
)
```

```python
from openLLV.deepLearning.models.LLIE.RetinexFormer import RetinexFormer

model = RetinexFormer(feature_channels=24, levels=2, clamp_output=False)
```

## Checkpoint / Official Weights

可将 openLLV 检查点直接传给 `llv.predict`。本实现不会自动获取或重映射官方 RetinexFormer 检查点。
