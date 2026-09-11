# Zero-DCE++

> 任务：低光图像增强（LLIE）

Zero-DCE++ 是基于深度可分离卷积的紧凑曲线估计网络，注册名为 `ZeroDCEPlusPlus`。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://ieeexplore.ieee.org/document/9369102/ |
| 官方源码 | https://github.com/Li-Chongyi/Zero-DCE_extension |
| 官方项目页 | https://li-chongyi.github.io/Proj_Zero-DCE++.html |
| 默认配置 | `openLLV/deepLearning/config/ZeroDCE++.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/ZeroDCEPlusPlus.py` |
| 类名 | `ZeroDCEPlusPlus` |
| 注册名 | `ZeroDCEPlusPlus`（无别名；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 关联损失 | `openLLV/deepLearning/loss/LLIELoss/ZeroDCE_Loss.py` 中的 `ZeroDCE_extension_Loss`（注册名 `zerodce_extension`；别名 `zerodceplusplus`、`zerodce++`） |

## Implementation Notes

`ZeroDCEPlusPlus(config=None, **kwargs)` 在 `config` 之后合并关键字覆盖值。七个深度可分离卷积块估计一个三通道曲线图，再执行八次二次更新。训练和推理的默认 `scale_factor` 均为 `12`。当其不为 `1` 时，先按 `1 / scale_factor` 缩放输入以估计曲线图，再使用 `align_corners=True` 的双线性插值将曲线图直接恢复到原始输入尺寸，保留官方上采样的对齐语义，同时支持尺寸不能被缩放因子整除的情况（包括 512 与缩放因子 12）。不裁剪图像。可选的卷积权重初始化使用 $N(0, 0.02)$，默认关闭。无参考损失基于增强图像计算曝光控制项，目标值为 `0.6`。训练返回含增强结果与曲线图的标准字典，推理返回张量。

内置 `ZeroDCE++.yaml` 设置 `model.params.scale_factor: 12` 和 `data.resize: [512, 512]`，使用数据集的抗锯齿双线性缩放，将训练及验证图像统一调整为 512×512。Adam 使用 `lr: 0.0001`、`weight_decay: 0.0001`；`scheduler.name: null` 保持学习率不变；`train.grad_clip: 0.1` 对全局梯度范数进行裁剪。这些默认值均可覆盖。因此该配置的训练缩放因子为 12，而官方训练脚本默认为 1。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 覆盖默认值的字典。 | 非字典且非 `None` 抛 `TypeError`；未知键保留但未被消费时不起作用。 |
| `**kwargs` | `Any` | `{}` | 在 `config` 后合并的覆盖值。 | 键语义与 `config` 相同。 |
| `model_name` | `str` | `"ZeroDCEPlusPlus"` | 共享模型元数据。 | 不校验，也不参与架构构造。 |
| `input_channels` | `int` | `3` | 第一深度卷积块接受的通道数。 | 必须为正整数；实际必须为 `3`，因为曲线输出与图像运算为三通道。 |
| `save_dir` | `str` | `"./checkpoints/llie/ZeroDCEPlusPlus"` | 默认检查点/配置输出目录。 | 构造时不校验。 |
| `number_f` | `int` | `32` | 曲线估计网络特征宽度。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `scale_factor` | `int \| float` | `12` | 训练和推理共同使用的输入下采样因子；曲线图恢复到输入的精确尺寸。 | 必须可比较且大于 `0`，否则抛 `ValueError`；缩放后的高、宽均须至少为 1，输入尺寸不必被缩放因子整除。 |
| `initialize_weights` | `bool` | `False` | 构造时是否将全部卷积权重按 $N(0, 0.02)$ 初始化。 | 必须为布尔值，否则抛 `TypeError`；偏置保留 PyTorch 默认初始化。 |
| `mode` | `str` | `"inference"` | 选择训练字典或推理张量输出契约。 | 只能是 `"train"` 或 `"inference"`，否则抛 `ValueError`。 |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroDCEPlusPlus",
    "input.jpg",
    output="results/zero-dce++/output.png",
    config={
        "scale_factor": 2,
        "initialize_weights": True,
        "mode": "inference",
    },
)
```

```python
import openLLV as llv

result = llv.train("ZeroDCEPlusPlus", root_dir="datasets/my_dataset")
```

## Checkpoint / Official Weights

将 openLLV `.pt` 或 `.pth` 检查点传给 `llv.predict`；显式 predictor 配置覆盖已保存值。未覆盖时，预测沿用模型/检查点中的缩放因子，包括旧检查点保存的 `1`；仅在未保存也未提供该参数时，才使用默认值 `12`。未实现官方权重下载器。
