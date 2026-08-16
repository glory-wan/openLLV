# BIMEF

> 文档分组：传统低光图像增强

BIMEF 使用对比度、饱和度和良好曝光度权重，融合原图与自动或手动曝光后的副本。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.48550/arXiv.1711.00591 |
| 官方源码 | 无 |
| 官方项目页 | 无 |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/LLIE/BIMEF.py` |
| 类名 | `BIMEF` |
| 注册名 | `BIMEF`（无别名；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

公共三通道输入和 NumPy 输出使用 RGB；BGR/BGRA 仅是 `_enhance()` 的内部工作布局。算法内部也接受灰度图，归一化至 `[0, 1]`，保留 alpha/通道布局并恢复源值域。`exposure_ratio=None` 时估计 `clip(target_mean / mean_luminance, 1, max_ratio)`。每个算法参数也可传给 `enhance()` 作为单次覆盖值；覆盖值会校验但不修改已存参数。工厂创建会过滤不支持的构造键并告警。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `exposure_ratio` | `Optional[float]` | `None` | 手动曝光乘数；`None` 启用估计。 | 非 `None` 时必须 `> 0`；可运行时覆盖。 |
| `target_mean` | `float` | `0.55` | 自动曝光目标平均亮度。 | 必须在 `(0, 1)`；可运行时覆盖。 |
| `max_ratio` | `float` | `5.0` | 自动曝光上限。 | 必须 `>= 1`；可运行时覆盖。 |
| `well_exposed_sigma` | `float` | `0.2` | 良好曝光权重的高斯 sigma。 | 必须 `> 0`；可运行时覆盖。 |
| `contrast_weight` | `float` | `1.0` | 对比度权重指数。 | 必须 `>= 0`；可运行时覆盖。 |
| `saturation_weight` | `float` | `1.0` | 饱和度权重指数。 | 必须 `>= 0`；可运行时覆盖。 |
| `well_exposed_weight` | `float` | `1.0` | 良好曝光度指数。 | 必须 `>= 0`；可运行时覆盖。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式。 | 其他值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 将输出转回输入 dtype。 | 非 `bool` 抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪到解析后的输入值域。 | 非 `bool` 抛 `TypeError`。 |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | 输入值域；自动区分通常的浮点 `[0,1]` 与 `[0,255]` 输入。 | 最大值 `<= 1` 的暗部字节值域浮点图须用 `"byte"`。自定义边界须有限且递增；图像值越界抛 `ValueError`。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("BIMEF", "input.jpg", output="results/bimef/output.png", exposure_ratio=2.0, target_mean=0.6)
```

```python
from openLLV.tradition.algorithms.LLIE.BIMEF import BIMEF
enhancer = BIMEF(max_ratio=4.0)
result = enhancer("input.jpg", contrast_weight=0.8)
```
