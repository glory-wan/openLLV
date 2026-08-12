# URetinex-Net

> 任务：低光图像增强（LLIE）

URetinex-Net 是结合 Retinex 分解、深度展开与照明调整的模型，在 openLLV 中注册为 `URetinexNet`。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| 官方源码 | https://github.com/AndersonYong/URetinex-Net |
| 官方项目页 | 无 |
| 默认配置 | `openLLV/deepLearning/config/URetinexNet.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/URetinex.py` |
| 类名 | `URetinexNet` |
| 注册名 | `URetinexNet`（无别名；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/URetinex_Loss.py` 中的 `URetinex_Loss`（注册名 `uretinex`；别名 `uretinex_loss`、`uretinexnet`、`uretinexnet_loss`、`URetinex_Loss`） |

## Implementation Notes

构造器为 `URetinexNet(config=None, **kwargs)`，关键字覆盖值优先于 `config`。每轮展开更新反射与照明代理，调整模块随后生成 `High_L`，预测为 `High_L * R`。传给 `forward` 的 `ratio` 优先级最高；否则，当 `adaptive_ratio=True` 且提供 `high_img` 时由第二个分解网络计算比率；再否则使用 `adjustment_ratio`。训练返回标准字典，`aux` 含反射、照明、调整后照明和所选比率；推理返回张量。`forward` 内部仅在配置 `mode="train"` 时启用梯度。架构假定三通道图像，`input_channels` 仅为元数据。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 覆盖模型默认值的字典。 | 非字典且非 `None` 会抛 `TypeError`；未知键会保留，但未被消费时不起作用。 |
| `**kwargs` | `Any` | `{}` | 在 `config` 之后合并的覆盖值。 | 键语义与 `config` 相同。 |
| `model_name` | `str` | `"URetinexNet"` | 共享模型元数据。 | 不校验，也不参与架构构造。 |
| `input_channels` | `int` | `3` | 共享输入通道元数据。 | 必须为正整数，否则抛 `ValueError`；实现仍假定 RGB 输入。 |
| `save_dir` | `str` | `"./checkpoints/llie/URetinexNet"` | 默认检查点/配置输出目录。 | 构造时不校验。 |
| `unfolding_rounds` | `int` | `3` | 展开迭代数。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `gamma` | `float` | `0.01` | 初始反射代理惩罚权重。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `lambda` | `float` | `0.01` | 初始照明代理惩罚权重。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `gamma_offset` | `float` | `0.01` | 第一轮之后每轮加入 `gamma` 的增量。 | 无显式校验。 |
| `lambda_offset` | `float` | `0.01` | 第一轮之后每轮加入 `lambda` 的增量。 | 无显式校验。 |
| `use_concat_l` | `bool` | `True` | 恢复模块是否将照明与反射特征拼接。 | 无显式校验；真值决定架构。 |
| `mode` | `str` | `"inference"` | 选择训练字典或推理张量输出。 | 只能是 `"train"` 或 `"inference"`，否则抛 `ValueError`。 |
| `adjustment_ratio` | `float` | `5.0` | 后备照明调整比率。 | 无显式校验。 |
| `adaptive_ratio` | `bool` | `False` | 构建并启用高亮图分解路径。 | 无显式校验；真值控制架构与前向路由。 |

`forward(x, high_img=None, ratio=None)` 还接受 `high_img: Optional[torch.Tensor]` 与 `ratio: Optional[float]`。它们是运行时输入而非配置；`ratio` 覆盖所有配置比率路径。

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("URetinexNet", "input.jpg", output="results/uretinex/output.png", config={"unfolding_rounds": 4, "adjustment_ratio": 4.0})
```

```python
import openLLV as llv

result = llv.train("URetinexNet", model_params={"mode": "train", "unfolding_rounds": 3})
```

## Checkpoint / Official Weights

将 openLLV `.pt` 或 `.pth` 检查点传给 `llv.predict`；显式 predictor 配置会覆盖已保存配置。未实现官方权重下载器。
