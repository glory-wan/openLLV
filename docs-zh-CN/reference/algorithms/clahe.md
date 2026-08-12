# CLAHE

> Documentation group: 基础方法

CLAHE 使用 OpenCV 执行限制对比度的局部直方图均衡。

## Links
| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 |
| 官方源代码 | None |
| 官方项目页 | None |

## Location in openLLV
| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/BaseMethods/CLAHE.py` |
| 类名 | `CLAHE` |
| 注册名 | `clahe`（类名也注册；无别名；查询忽略大小写） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

灰度图直接均衡。BGR 输入中 `rgb` 处理所有通道，其他模式处理相应亮度/明度通道。非 `uint8` 输入先转换。按次算法参数被忽略。OpenCV 对象在构造时缓存，`set_params` 改变 `clip_limit` 或 `tile_grid_size` 不会重建它，应新建增强器。

## Parameters
| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | `rgb`/`bgr`、`hsv`、`hls`、`yuv`/`ycbcr` 或 `lab`；忽略大小写。非法类型/值抛 `TypeError`/`ValueError`。 |
| `clip_limit` | `float` | `2.0` | 有限、非布尔、数值且大于零，否则抛 `ValueError`。 |
| `tile_grid_size` | `Tuple[int, int]` | `(8, 8)` | 恰含两个正的非布尔整数，否则抛 `ValueError`。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式；非法值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 保留输入 dtype；非布尔值抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪至有效 dtype 范围；非布尔值抛 `TypeError`。 |

## Usage Example
```python
import openLLV as llv
enhanced, saved_path = llv.predict("clahe", "input.jpg", output="results/clahe.png", color_space="hsv", clip_limit=3.0, tile_grid_size=(4, 4))
```
