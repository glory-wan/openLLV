# LIME

> 文档分组：传统低光图像增强

LIME 从逐像素通道最大值估计照明，以引导滤波细化，再用图像除以 gamma 调整后的照明。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TIP.2016.2639450 |
| 官方源码 | 无 |
| 官方项目页 | 无 |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/LLIE/LIME.py` |
| 类名 | `LIME` |
| 注册名 | `LIME`（无别名；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

LIME 支持 BGR、灰度和 BGRA 数组并保留灰度/alpha 布局。整数输入会归一化并恢复 dtype 值域，浮点输入裁剪至 `[0, 1]`。每个算法参数均可传给 `enhance()` 单次覆盖，经校验且不修改已存状态。工厂创建会过滤不支持的构造键并告警。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `gamma` | `float` | `0.8` | 施加于细化照明的指数。 | 必须 `> 0`；可运行时覆盖。 |
| `guided_radius` | `int` | `15` | 引导滤波窗口尺寸。 | 必须 `> 0`；可运行时覆盖。 |
| `guided_eps` | `float` | `0.001` | 引导滤波正则项。 | 必须 `> 0`；可运行时覆盖。 |
| `illumination_floor` | `float` | `0.05` | 除法前的照明下限。 | 必须在 `(0, 1]`；可运行时覆盖。 |
| `exposure` | `float` | `1.0` | 照明校正后的全局乘数。 | 必须 `> 0`；可运行时覆盖。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式。 | 其他值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 将输出转回输入 dtype。 | 非 `bool` 抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪到目标 dtype 有效范围。 | 非 `bool` 抛 `TypeError`。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("LIME", "input.jpg", output="results/lime/output.png", gamma=0.9, exposure=1.1)
```

```python
from openLLV.tradition.algorithms.LLIE.LIME import LIME
enhancer = LIME(guided_radius=21)
result = enhancer("input.jpg", illumination_floor=0.08)
```
