# LLNet

> 任务：低光图像增强与去噪（LLIE）

LLNet 是在 openLLV 中注册的重叠图块全连接自编码器，用于低光增强与去噪。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1016/j.patcog.2016.06.008 |
| 官方源码 | https://github.com/kglore/llnet_color |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/LLNet.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/LLNet.py` |
| 类名 | `LLNet` |
| 注册名 | `LLNet`（无别名） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/LLNet_Loss.py`（`llnet`；别名：`llnet_loss`、`LLNet_Loss`、`sparse_denoising_autoencoder`） |

## Implementation Notes

LLNet 对输入作反射填充并提取重叠图块，将展平图块送入对称的全连接编码器/解码器，折叠回图像后除以重叠计数，再裁回原尺寸。训练模式返回含隐藏激活和全部正则化权重的 `{"pred", "aux", "meta"}`；推理返回增强张量。较大的默认值 `hidden_dims=[2000, 1600, 1200]` 会占用大量内存。

构造器为 `LLNet(config=None, **kwargs)`，依次合并默认值、`config` 和关键字覆盖。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置字典。 | 必须是字典或 `None`。 |
| `model_name` | `str` | `"LLNet"` | 继承自 `LLVModel` 的模型名。 | 无额外校验。 |
| `input_channels` | `int` | `3` | 输入通道数及图块向量通道数。 | 必须为正整数。 |
| `save_dir` | `str` | `"./checkpoints/llie/LLNet"` | 默认检查点/配置目录。 | 无额外校验。 |
| `patch_size` | `int` | `17` | 正方形图块边长。 | 经 `int()` 转换后必须为正奇数；运行时反射填充还要求输入足够大。 |
| `patch_stride` | `int` | `3` | 图块提取和折叠步长。 | 经 `int()` 转换后必须为正数。 |
| `hidden_dims` | `Union[List[int], Tuple[int, ...]]` | `[2000, 1600, 1200]` | 编码器宽度；解码器反向使用。 | 必须是非空列表/元组，且各值可转为正整数。 |
| `activation` | `str` | `"sigmoid"` | 隐藏层激活函数。 | 只能是 `"sigmoid"`、`"relu"` 或 `"tanh"`。 |
| `output_activation` | `str` | `"sigmoid"` | 最终图块激活。 | 只能是 `"sigmoid"`、`"clamp"`（截断至 `[0, 1]`）或 `"none"`。 |
| `mode` | `str` | `"inference"` | 选择结构化训练输出或张量推理输出。 | 只能是 `"train"` 或 `"inference"`。 |

所有配置键均可经构造器 `**kwargs` 提供并覆盖 `config`；未知键会保留但不使用。

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LLNet",
    "input.jpg",
    output="results/llnet/output.png",
    config={"patch_stride": 2, "output_activation": "clamp"},
)
```

```python
from openLLV.deepLearning.models.LLIE.LLNet import LLNet

model = LLNet(patch_size=9, hidden_dims=[512, 256], activation="relu")
```

## Checkpoint / Official Weights

openLLV 检查点可直接传给 `llv.predict`。本实现不会自动下载或转换官方仓库权重。
