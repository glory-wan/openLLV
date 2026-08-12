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

`ZeroDCEPlusPlus(config=None, **kwargs)` 在 `config` 之后合并关键字覆盖值。七个深度可分离卷积块估计一个三通道曲线图，再执行八次二次更新。仅在推理中，`scale_factor > 1` 会先下采样图像进行曲线估计，再上采样曲线图。训练不下采样，却仍在 `scale_factor > 1` 时上采样曲线图，会使其空间尺寸与原图不兼容，因此训练应使用 `scale_factor=1`。训练返回含增强结果与曲线图的标准字典，推理返回张量。`train_mode()` 与 `eval_mode()` 同时更新配置模式和下采样模块。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 覆盖默认值的字典。 | 非字典且非 `None` 抛 `TypeError`；未知键保留但未被消费时不起作用。 |
| `**kwargs` | `Any` | `{}` | 在 `config` 后合并的覆盖值。 | 键语义与 `config` 相同。 |
| `model_name` | `str` | `"ZeroDCEPlusPlus"` | 共享模型元数据。 | 不校验，也不参与架构构造。 |
| `input_channels` | `int` | `3` | 第一深度卷积块接受的通道数。 | 必须为正整数；实际必须为 `3`，因为曲线输出与图像运算为三通道。 |
| `save_dir` | `str` | `"./checkpoints/llie/ZeroDCEPlusPlus"` | 默认检查点/配置输出目录。 | 构造时不校验。 |
| `number_f` | `int` | `32` | 曲线估计网络特征宽度。 | 必须可比较且大于 `0`，否则抛 `ValueError`。 |
| `scale_factor` | `int | float` | `1` | 推理时曲线估计下采样与曲线图上采样倍数。 | 必须可比较且大于 `0`，否则抛 `ValueError`；训练须使用 `1` 以保持空间尺寸兼容。 |
| `mode` | `str` | `"inference"` | 选择架构路由与输出契约。 | 只能是 `"train"` 或 `"inference"`，否则抛 `ValueError`。 |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict("ZeroDCEPlusPlus", "input.jpg", output="results/zero-dce++/output.png", config={"scale_factor": 2, "mode": "inference"})
```

```python
import openLLV as llv

result = llv.train("ZeroDCEPlusPlus", model_params={"mode": "train", "scale_factor": 1})
```

## Checkpoint / Official Weights

将 openLLV `.pt` 或 `.pth` 检查点传给 `llv.predict`；显式 predictor 配置覆盖已保存值。未实现官方权重下载器。
