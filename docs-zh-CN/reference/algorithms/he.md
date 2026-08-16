# HE

> Documentation group: 基础方法

HE 对灰度图、内部转换后的各 BGR 通道或指定亮度通道执行全局直方图均衡；其公共 NumPy 契约为 RGB。

## Links
| 类型 | URL |
| --- | --- |
| 论文 | None |
| 官方源代码 | None |
| 官方项目页 | None |

## Location in openLLV
| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/BaseMethods/HE.py` |
| 类名 | `HE` |
| 注册名 | `he`（类名也注册；无别名；查询忽略大小写） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

公共三通道输入和 NumPy 输出使用 RGB；`LLVEnhancer` 在 `_enhance()` 前转为 BGR，结束后转回 RGB。输入先转为 OpenCV 所需的 `uint8`。灰度图直接均衡；内部 `rgb` 表示逐 BGR 通道均衡，`hsv`、`hls`、`yuv`、`lab` 分别处理 V、L、Y、L。按次算法参数被忽略。

## Parameters
| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"rgb"` | `rgb`/`bgr`、`hsv`、`hls`、`yuv`/`ycbcr` 或 `lab`；非法类型/值抛 `TypeError`/`ValueError`。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式；非法值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 保留输入 dtype；非布尔值抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪至解析后的输入值域；非布尔值抛 `TypeError`。 |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | 输入值域。自动推断浮点 `[0,1]` 或 `[0,255]`；最大值 `<= 1` 的暗部字节值域浮点图须显式用 `"byte"`。自定义边界必须有限且递增；图像值越界抛 `ValueError`。 |

## Usage Example
```python
import openLLV as llv
enhanced, saved_path = llv.predict("he", "input.jpg", output="results/he.png", color_space="lab")
```
