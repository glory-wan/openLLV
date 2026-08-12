# RCLAHE

> Documentation group: 基础方法

RCLAHE 递归应用相同 CLAHE 操作以增强局部对比度。

## Links
| 类型 | URL |
| --- | --- |
| 论文 | None |
| 官方源代码 | None |
| 官方项目页 | None |

## Location in openLLV
| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/BaseMethods/RCLAHE.py` |
| 类名 | `RCLAHE` |
| 注册名 | `rclahe`（类名也注册；无别名；查询忽略大小写） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

输入转为 `uint8` 后重复 CLAHE `iterations` 次，通道行为同 CLAHE。按次算法参数被忽略。`set_params` 不重建 OpenCV 对象，改变 `clip_limit` 或 `tile_grid_size` 后应重建实例；改变 `iterations` 会生效。

## Parameters
| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | `rgb`/`bgr`、`hsv`、`hls`、`yuv`/`ycbcr` 或 `lab`；非法类型/值抛 `TypeError`/`ValueError`。 |
| `clip_limit` | `float` | `2.0` | 有限、非布尔、数值且大于零，否则抛 `ValueError`。 |
| `tile_grid_size` | `Tuple[int, int]` | `(8, 8)` | 恰含两个正的非布尔整数，否则抛 `ValueError`。 |
| `iterations` | `int` | `3` | 正的非布尔整数，否则抛 `ValueError`。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式；非法值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 保留输入 dtype；非布尔值抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪至有效 dtype 范围；非布尔值抛 `TypeError`。 |

## Usage Example
```python
import openLLV as llv
enhanced, saved_path = llv.predict("rclahe", "input.jpg", output="results/rclahe.png", clip_limit=2.5, iterations=2)
```
