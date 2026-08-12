# Zero-DCE

> 任务：低光图像增强（LLIE）

Zero-DCE 估计逐像素曲线参数并执行八次二次增强更新，在 openLLV 中注册为 `ZeroDCE`。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf |
| 官方源码 | https://github.com/Li-Chongyi/Zero-DCE |
| 官方项目页 | https://li-chongyi.github.io/Proj_Zero-DCE.html |
| 默认配置 | `openLLV/deepLearning/config/ZeroDCE.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/ZeroDCE.py` |
| 类名 | `ZeroDCE` |
| 注册名 | `ZeroDCE`（无别名；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/ZeroDCE_Loss.py` 中的 `ZeroDCE_Loss`（注册名 `zerodce`；无别名） |

## Implementation Notes

`ZeroDCE(config=None, **kwargs)` 在 `config` 之后合并关键字覆盖值。七个卷积层估计曲线图。尽管 `num_iterations` 控制末层通道数，`forward` 无条件将结果拆成恰好八个三通道图并执行八次更新。因此可工作的配置是 `num_iterations=8`；其他正值能通过构造校验，但会在前向中失败或产生不兼容行为。训练返回含 `pred`、`aux.enhanced` 与 `aux.r` 的标准字典，推理返回增强张量。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 覆盖默认值的字典。 | 非字典且非 `None` 抛 `TypeError`；未知键保留但未被消费时不起作用。 |
| `**kwargs` | `Any` | `{}` | 在 `config` 后合并的覆盖值。 | 键语义与 `config` 相同。 |
| `model_name` | `str` | `"ZeroDCE"` | 共享模型元数据。 | 不校验，也不参与架构构造。 |
| `input_channels` | `int` | `3` | 第一层卷积输入通道数。 | 必须为正整数；实际必须为 `3`，因为曲线图与更新运算均为三通道。 |
| `save_dir` | `str` | `"./checkpoints/llie/ZeroDCE"` | 默认检查点/配置输出目录。 | 构造时不校验。 |
| `number_f` | `int` | `32` | 曲线估计网络特征宽度。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `num_iterations` | `int` | `8` | 决定输出曲线图通道数的乘数。 | 必须可比较且大于 `0`；由于前向固定解包八个图，实际必须为 `8`。 |
| `mode` | `str` | `"inference"` | 选择训练字典或推理张量输出。 | 只能是 `"train"` 或 `"inference"`，否则抛 `ValueError`。 |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("ZeroDCE", "input.jpg", output="results/zero-dce/output.png", config={"number_f": 24, "num_iterations": 8})
```

```python
import openLLV as llv

result = llv.train("ZeroDCE", root_dir="data/LOL-v1")
```

## Checkpoint / Official Weights

将 openLLV `.pt` 或 `.pth` 检查点传给 `llv.predict`；显式 predictor 配置覆盖已保存配置。未实现官方权重下载器。
