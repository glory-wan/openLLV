# DarkChannel（DCP）

> 文档分组：传统去雾 / 低光图像增强

DarkChannel 对反相输入应用暗通道先验，再将恢复图反相，得到低光增强结果。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://ieeexplore.ieee.org/document/5206515 |
| 官方源码 | 无 |
| 官方项目页 | 无 |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/Dehazing/DCP.py` |
| 类名 | `DarkChannel` |
| 注册名 | `DarkChannel`（别名：`dcp`；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

实现期望 8 位值域的三通道 BGR 图像，在反相图上估计大气光与透射率，以灰度引导滤波细化透射率，并返回按 `255` 缩放的值。与本组其他算法不同，`_enhance()` 忽略全部运行时关键字；请在构造时或通过 `set_params()` 修改算法值。工厂创建会过滤不支持的构造键并发出 `UserWarning`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `size` | `int` | `15` | 矩形暗通道腐蚀核尺寸。 | 必须 `> 0`。 |
| `omega` | `float` | `0.95` | 透射率估计权重。 | 必须在 `(0, 1]`。 |
| `t_min` | `float` | `0.1` | 恢复时的透射率下限。 | 必须在 `(0, 1)`。 |
| `guided_radius` | `int` | `60` | 引导滤波窗口尺寸。 | 必须 `> 0`。 |
| `guided_eps` | `float` | `0.0001` | 引导滤波正则项。 | 必须 `> 0`。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式。 | 其他值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 将输出转回输入 dtype。 | 非 `bool` 抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪到目标 dtype 有效范围。 | 非 `bool` 抛 `TypeError`。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("dcp", "input.jpg", output="results/dcp/output.png", size=9, omega=0.9)
```

```python
from openLLV.tradition.algorithms.Dehazing.DCP import DarkChannel
enhancer = DarkChannel(t_min=0.15)
enhancer.set_params(guided_radius=31)
result = enhancer("input.jpg")
```
