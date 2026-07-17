# HE

> 文档分组：基础方法

HE 是 openLLV 已实现的传统直方图均衡算法。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 / 经典来源 | https://doi.org/10.1016/S0146-664X(77)80011-7 |
| 相关早期文献 | https://doi.org/10.1109/T-C.1974.223892 |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

HE（Histogram Equalization，直方图均衡）是一种经典的全局对比度增强方法。它计算图像的灰度分布并构建累积分布函数，将原始灰度级重映射为更均匀的强度分布，从而提升图像的整体对比度。

在低光增强中，HE 可以提高暗部区域的可见性，但也可能造成局部过度增强、噪声放大或色偏。因此，对于彩色图像，通常需要选择合适的颜色空间，并且只对亮度通道执行均衡。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/BaseMethods/HE.py` |
| 算法类名 | `HE` |
| 注册名称 | `he` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 实现说明

openLLV 的 HE 支持灰度图像和多种颜色空间：

- `rgb` / `bgr`
- `hsv`
- `hls`
- `yuv` / `ycbcr`
- `lab`

选择与亮度相关的颜色空间时，算法会均衡亮度通道，然后转换回 BGR 图像。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"rgb"` | 执行直方图均衡的颜色空间 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "he",
    "input.jpg",
    output="results/he/output.jpg",
    color_space="hsv",
)
```

文件夹批处理：

```python
saved_paths = llv.predict(
    "he",
    "images/",
    output="results/he",
    color_space="yuv",
)
```

