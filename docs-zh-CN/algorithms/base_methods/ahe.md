# AHE

> 文档分组：基础方法

AHE 是 openLLV 已实现的自适应直方图均衡算法。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 / 经典来源 | https://doi.org/10.1016/S0734-189X(87)80186-X |
| 论文页面 | https://www.sciencedirect.com/science/article/abs/pii/S0734189X8780186X |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

AHE（Adaptive Histogram Equalization，自适应直方图均衡）是 HE 的局部版本。它不会在整张图像上执行一次全局直方图均衡，而是在局部区域内计算直方图映射，因此能够增强局部细节和局部对比度。

AHE 的主要问题是容易放大平坦区域中的噪声。后来 CLAHE 通过限制局部直方图的裁剪阈值缓解了这一问题。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/BaseMethods/AHE.py` |
| 算法类名 | `AHE` |
| 注册名称 | `ahe` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 实现说明

openLLV 的 AHE 使用 OpenCV 的 CLAHE 接口，并设置较大的 `clipLimit` 来近似 AHE 行为。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | 执行均衡化的颜色空间 |
| `tile_grid_size` | `tuple` | `(8, 8)` | 局部网格大小 |

支持的颜色空间：

- `rgb` / `bgr`
- `hsv`
- `hls`
- `yuv` / `ycbcr`
- `lab`

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ahe",
    "input.jpg",
    output="results/ahe/output.jpg",
    color_space="yuv",
    tile_grid_size=(8, 8),
)
```

