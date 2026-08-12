# DarkIR

> Task: 低光图像恢复（LLIE）

DarkIR 是 openLLV 中联合执行低光增强与恢复的编码器—解码器模型。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2025/papers/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.pdf |
| 官方源代码 | https://github.com/cidautai/DarkIR |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/DarkIR.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/DarkIR.py` |
| 类名 | `DarkIR` |
| 注册名 | `DarkIR`（无别名；查询忽略大小写及首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/DarkIR_Loss.py` |

## Implementation Notes

DarkIR 将空间尺寸零填充至 `2 ** len(enc_blk_nums)` 的倍数，并把主残差输出裁剪回输入尺寸。训练模式返回标准字典；仅启用辅助损失时存在 `aux.side_output`，且其分辨率为瓶颈分辨率。前向传入 `side_loss=True` 会持久地设置 `model.config["side_loss"] = True`。YAML 设为 `side_loss: true`，直接构造则默认为 `False`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置映射；非字典抛 `TypeError`；`**kwargs` 优先。 |
| `model_name` | `str` | `"DarkIR"` | 基类元数据。 |
| `input_channels` | `int` | `3` | 图像通道；须为正整数。 |
| `save_dir` | `str` | `"./checkpoints/llie/DarkIR"` | 默认检查点目录。 |
| `width` | `int` | `32` | 初始特征宽度；须为正。 |
| `middle_blk_num_enc` | `int` | `2` | 编码中间块数；须为正。 |
| `middle_blk_num_dec` | `int` | `2` | 解码中间块数；须为正。 |
| `enc_blk_nums` | `List[int]` | `[1, 2, 3]` | 各编码层块数，并决定填充倍数；元素无显式校验。 |
| `dec_blk_nums` | `List[int]` | `[3, 1, 1]` | 各解码层块数；应与编码深度匹配以完整遍历跳连/上采样，但无显式校验。 |
| `dilations` | `List[int]` | `[1, 4, 9]` | 传给每个解码块的膨胀率；无显式校验。 |
| `extra_depth_wise` | `bool` | `True` | 启用编解码块中的额外深度卷积分支。 |
| `mode` | `str` | `"inference"` | 须为 `"train"` 或 `"inference"`。 |
| `side_loss` | `bool` | `False` | 启用瓶颈辅助输出；YAML 覆盖为 `True`，前向 `side_loss=True` 也会持久启用。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("DarkIR", "input.jpg", output="results/darkir/output.png", config={"width": 32})
```

## Checkpoint / Official Weights

将 openLLV 检查点路径作为预测目标。未实现官方权重自动下载。
