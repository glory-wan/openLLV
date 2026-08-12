# RUAS

> 任务：无监督低光图像增强（LLIE）

RUAS 是具有搜索所得照明估计和去噪网络的 Retinex 启发式展开模型。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf |
| 官方源码 | https://github.com/KarelZhang/RUAS |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/RUAS.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/RUAS.py` |
| 类名 | `RUAS` |
| 注册名 | `RUAS`（无别名） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/RUAS_Loss.py`（`ruas`；无别名） |

## Implementation Notes

增强网络迭代估计截断后的透射图并计算 `input / transmission`；去噪网络从最终增强图中减去预测噪声。固定的 `IEM_Genotype` 和 `NRM_Genotype` 操作图定义两个搜索网络。训练模式下，`forward()` 返回 `{"pred", "aux", "meta"}`，包含所有增强图、所有透射图及噪声。推理返回最终去噪图，不执行显式的最终截断。

若 `pretrained_denoise_path` 为真，初始化会调用 `torch.load(..., map_location="cpu")` 并将结果直接载入 `denoise_net`。所有异常都会被捕获并打印，因此文件缺失或不兼容不会中止构造。构造器为 `RUAS(config=None, **kwargs)`，关键字值覆盖 `config`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置字典。 | 必须是字典或 `None`。 |
| `model_name` | `str` | `"RUAS"` | 继承自 `LLVModel` 的模型名。 | 无额外校验。 |
| `input_channels` | `int` | `3` | 基类模型的输入通道元数据。 | 必须是正整数；架构本身固定为三通道，修改此键不会重建这些固定层。 |
| `save_dir` | `str` | `"./checkpoints/llie/RUAS"` | 默认检查点/配置目录。 | 无额外校验。 |
| `iem_nums` | `int` | `3` | 照明估计迭代次数。 | 直接与零比较且必须为正数；随后供 `range()` 使用，因此须兼容整数。 |
| `nrm_nums` | `int` | `3` | 去噪搜索块数量。 | 直接与零比较且必须为正数；随后供 `range()` 使用。 |
| `enhance_channel` | `int` | `3` | 增强搜索块特征宽度。 | 无显式校验；必须是合法正卷积通道数。固定基因型含残差/恒等操作，不兼容宽度会在构造或前向时报错。 |
| `denoise_channel` | `int` | `6` | 去噪搜索块特征宽度。 | 无显式校验；必须是合法正卷积通道数。 |
| `mode` | `str` | `"inference"` | 选择结构化训练输出或张量推理输出。 | 只能是 `"train"` 或 `"inference"`。 |
| `pretrained_denoise_path` | `Optional[Union[str, pathlib.Path]]` | `None` | 构造时加载的去噪网络原始状态字典路径。 | 任意真值触发加载；错误会被捕获并打印而非抛出。 |

所有配置键均可经构造器 `**kwargs` 传入；未知键会保留但不使用。前向 `**kwargs` 会被接受并忽略。

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "RUAS",
    "input.jpg",
    output="results/ruas/output.png",
    config={"iem_nums": 4, "nrm_nums": 2},
)
```

```python
from openLLV.deepLearning.models.LLIE.RUAS import RUAS

model = RUAS(pretrained_denoise_path="weights/ruas_denoise.pt")
```

## Checkpoint / Official Weights

将 openLLV 完整模型检查点传给 `llv.predict` 使用。`pretrained_denoise_path` 不同：它要求 `denoise_net` 自身的状态字典，而不是 openLLV 检查点包装。官方权重不会自动下载。
