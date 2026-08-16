# AHE

> Documentation group: 基础方法

AHE 通过固定 `clipLimit=255.0` 的 OpenCV CLAHE 近似自适应直方图均衡。

## Links
| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1016/S0734-189X(87)80186-X |
| 官方源代码 | None |
| 官方项目页 | None |

## Location in openLLV
| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/BaseMethods/AHE.py` |
| 类名 | `AHE` |
| 注册名 | `ahe`（类名也注册；无别名；查询忽略大小写及首尾空白） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

公共三通道输入和 NumPy 输出使用 RGB。`LLVEnhancer` 在 `_enhance()` 前把 RGB 转为内部 BGR，并在结束后把结果转回 RGB。灰度图直接均衡；内部 `rgb` 模式均衡所有 BGR 通道，其他空间分别均衡 V、L、Y 或 L 并转回 BGR。非 `uint8` 数据先转为 `uint8`。按次算法参数被忽略。`set_params` 不会重建缓存的 OpenCV CLAHE 对象，因此要有效改变 `tile_grid_size` 应新建实例。

## Parameters
| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | `rgb`、`bgr`（`rgb` 别名）、`hsv`、`hls`、`yuv`、`ycbcr`（`yuv` 别名）或 `lab`；去空白且忽略大小写。非法类型/值抛 `TypeError`/`ValueError`。 |
| `tile_grid_size` | `Tuple[int, int]` | `(8, 8)` | 恰含两个正的非布尔整数的元组/列表，否则抛 `ValueError`。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式；非法值抛 `ValueError`。传统预测器强制实例为 `"numpy"`。 |
| `keep_dtype` | `bool` | `True` | 转回输入 dtype；非布尔值抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 转换前裁剪到解析后的输入值域；非布尔值抛 `TypeError`。 |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | 输入值域。自动推断浮点 `[0,1]` 或 `[0,255]`；最大值 `<= 1` 的暗部字节值域浮点图须显式用 `"byte"`。自定义边界必须有限且递增；图像值越界抛 `ValueError`。 |

## Usage Example
```python
import openLLV as llv
enhanced, saved_path = llv.predict("ahe", "input.jpg", output="results/ahe.png", color_space="lab", tile_grid_size=(4, 4))
```
