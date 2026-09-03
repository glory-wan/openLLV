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

公共三通道输入和 NumPy 输出使用 RGB；LIME 仅把 BGR/BGRA 作为 `_enhance()` 内部布局，并在内部支持灰度图。它会保留灰度/alpha 布局。基类将解析后的源值域映射到 LIME 的 `[0,1]` 工作值域，随后恢复原输入的值域约定与可选 dtype。每个算法参数均可传给 `enhance()` 单次覆盖，经校验且不修改已存状态。工厂会忽略不支持的构造键，并在控制台说明其不会被使用且不影响计算；近似拼写会获得建议。

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
| `clip_output` | `bool` | `True` | 裁剪到解析后的输入值域。 | 非 `bool` 抛 `TypeError`。 |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | 输入值域；自动区分通常的浮点 `[0,1]` 与 `[0,255]` 输入。 | 最大值 `<= 1` 的暗部字节值域浮点图须用 `"byte"`。自定义边界须有限且递增；图像值越界抛 `ValueError`。 |

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
