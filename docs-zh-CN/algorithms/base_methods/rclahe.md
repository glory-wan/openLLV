# RCLAHE

> 文档分组：基础方法

RCLAHE 是 openLLV 已实现的递归 CLAHE 增强算法。

## 相关链接

| 类型 | URL |
| --- | --- |
| 基础算法论文 / 经典来源 | https://ieeexplore.ieee.org/document/109340/ |
| CLAHE Graphics Gems 章节 | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

RCLAHE（Recursive CLAHE）是基于 CLAHE 的递归增强版本。它反复对输入图像应用 CLAHE，经过多次处理逐步增强局部对比度。

与单次 CLAHE 相比，RCLAHE 可以获得更强的局部增强效果，但迭代次数过多可能导致过度增强、噪声放大或颜色不自然。因此，在实际使用中应控制 `iterations` 和 `clip_limit`。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/BaseMethods/RCLAHE.py` |
| 算法类名 | `RCLAHE` |
| 注册名称 | `rclahe` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 实现说明

openLLV 的 RCLAHE 会在 `_enhance()` 内部重复调用 CLAHE 处理流水线。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | 执行 CLAHE 的颜色空间 |
| `clip_limit` | `float` | `2.0` | 对比度裁剪阈值 |
| `tile_grid_size` | `tuple` | `(8, 8)` | 局部网格大小 |
| `iterations` | `int` | `2` | 递归应用 CLAHE 的次数 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "rclahe",
    "input.jpg",
    output="results/rclahe/output.jpg",
    color_space="hsv",
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    iterations=3,
)
```

