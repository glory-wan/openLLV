# MSRCR

> Documentation group: Retinex 低光增强

MSRCR 在多尺度 Retinex 上加入逐通道颜色恢复及全局增益/偏移。

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
| 类名 | `MSRCR` |
| 注册名 | `MSRCR`（无别名；查询忽略大小写） |
| 基类 | `MSR` → `_RetinexBase` → `LLVEnhancer` |

## Implementation Notes

MSR 后，彩色输入使用 `beta * (log(alpha*channel+eps)-log(channel_sum+eps))`，恢复响应为 `gain * (color_restoration * retinex + offset)`；灰度使用 `gain * (retinex + offset)`。输出按百分位归一化并保留 alpha。所有 MSRCR 专有值及 `scales` 均支持按次覆盖，不修改实例。

## Parameters
| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | 非空且元素可转为正数的可迭代对象；支持按次覆盖。 |
| `alpha` | `float` | `125.0` | 颜色恢复强度增益；须为数值且大于零。支持按次覆盖。 |
| `beta` | `float` | `46.0` | 颜色恢复对数增益；须为数值且大于零。支持按次覆盖。 |
| `gain` | `float` | `1.0` | 全局增益；须为数值且大于零。支持按次覆盖。 |
| `offset` | `float` | `0.0` | 全局偏移；须为数值，无范围约束。支持按次覆盖。 |
| `low_clip` | `float` | `1.0` | 下百分位；须满足 `0 <= low_clip < high_clip <= 100`。 |
| `high_clip` | `float` | `99.0` | 上百分位；约束同上。 |
| `eps` | `float` | `1e-6` | 稳定项；须大于零。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式；非法值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 保留输入 dtype；非布尔值抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪至有效 dtype 范围；非布尔值抛 `TypeError`。 |

## Usage Example
```python
import openLLV as llv
enhanced, saved_path = llv.predict("MSRCR", "input.jpg", output="results/msrcr.png", scales=(15.0, 100.0), alpha=100.0, beta=40.0)
```
