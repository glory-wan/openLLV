# MSR

> Documentation group: Retinex 低光增强

MSR 对多个高斯环绕尺度的对数域 Retinex 响应取平均。

## Links
| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/83.597272 |
| 官方源代码 | None |
| 官方项目页 | None |

## Location in openLLV
| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/LLIE/Retinex.py` |
| 类名 | `MSR` |
| 注册名 | `MSR`（无别名；查询忽略大小写） |
| 基类 | `_RetinexBase`，再到 `LLVEnhancer` |

## Implementation Notes

公共三通道输入和 NumPy 输出使用 RGB；基类在 `_enhance()` 边界完成内部 BGR 往返转换。每个尺度产生 SSR 响应，取平均后逐通道按百分位归一化。保留灰度与 alpha 布局。`scales` 可按次覆盖而不修改配置元组。

## Parameters
| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | 非字符串、非空且元素可转为正数的可迭代对象；存为浮点元组。非法可迭代/类型抛 `TypeError`，空或非正抛 `ValueError`。支持按次覆盖。 |
| `low_clip` | `float` | `1.0` | 下百分位；须满足 `0 <= low_clip < high_clip <= 100`。 |
| `high_clip` | `float` | `99.0` | 上百分位；约束同上。 |
| `eps` | `float` | `1e-6` | 稳定项；须大于零。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式；非法值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 保留输入 dtype；非布尔值抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪至解析后的输入值域；非布尔值抛 `TypeError`。 |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | 输入值域。自动推断浮点 `[0,1]` 或 `[0,255]`；最大值 `<= 1` 的暗部字节值域浮点图须显式用 `"byte"`。自定义边界必须有限且递增；图像值越界抛 `ValueError`。 |

## Usage Example
```python
import openLLV as llv
enhanced, saved_path = llv.predict("MSR", "input.jpg", output="results/msr.png", scales=(10.0, 60.0, 180.0))
```
