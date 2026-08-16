# SSR

> Documentation group: Retinex 低光增强

SSR 计算对数域单尺度 Retinex 响应，并以百分位归一化用于显示。

## Links
| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/83.557356 |
| 官方源代码 | None |
| 官方项目页 | None |

## Location in openLLV
| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/LLIE/Retinex.py` |
| 类名 | `SSR` |
| 注册名 | `SSR`（无别名；查询忽略大小写及首尾空白） |
| 基类 | `_RetinexBase`，再到 `LLVEnhancer`（位于 `Retinex.py` / `BaseModel.py`） |

## Implementation Notes

公共三通道输入和 NumPy 输出使用 RGB；基类在 `_enhance()` 边界完成内部 BGR 往返转换。基类将解析后的源值域映射到 SSR 的 `[0,1]` 工作值域，随后恢复原输入的值域约定与可选 dtype。响应为 `log(image+eps)-log(GaussianBlur(image)+eps)`，逐通道按配置百分位归一化。保留灰度布局和第四个 alpha 通道。`sigma` 可按次覆盖而不修改实例。

## Parameters
| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `sigma` | `float` | `80.0` | 高斯环绕尺度；须为数值且大于零（Python 数值检查会接受 `bool`）。支持按次覆盖。 |
| `low_clip` | `float` | `1.0` | 归一化下百分位；与 `high_clip` 须满足 `0 <= low_clip < high_clip <= 100`。 |
| `high_clip` | `float` | `99.0` | 归一化上百分位；约束同上。 |
| `eps` | `float` | `1e-6` | 对数/除法稳定项；须大于零。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式；非法值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 保留输入 dtype；非布尔值抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪至解析后的输入值域；非布尔值抛 `TypeError`。 |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | 输入值域。自动推断浮点 `[0,1]` 或 `[0,255]`；最大值 `<= 1` 的暗部字节值域浮点图须显式用 `"byte"`。自定义边界必须有限且递增；图像值越界抛 `ValueError`。 |

## Usage Example
```python
import openLLV as llv
enhanced, saved_path = llv.predict("SSR", "input.jpg", output="results/ssr.png", sigma=100.0, low_clip=0.5, high_clip=99.5)
```
