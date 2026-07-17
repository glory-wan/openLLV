# CLAHE

> 文档分组：基础方法

CLAHE 是 openLLV 已实现的限制对比度自适应直方图均衡算法。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 / 经典来源 | https://ieeexplore.ieee.org/document/109340/ |
| Graphics Gems 章节 | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

CLAHE（Contrast Limited Adaptive Histogram Equalization，限制对比度自适应直方图均衡）是 AHE 的改进版本。它在局部直方图均衡的基础上引入裁剪阈值，以限制局部对比度的放大，从而减轻噪声过度增强的问题。

CLAHE 常用于医学图像、低光图像、遥感图像以及其他低对比度图像增强任务。与 HE 相比，CLAHE 更关注局部细节；与 AHE 相比，CLAHE 对噪声更加稳健。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/BaseMethods/CLAHE.py` |
| 算法类名 | `CLAHE` |
| 注册名称 | `clahe` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 实现说明

openLLV 的 CLAHE 基于 OpenCV 的 `cv2.createCLAHE()`，支持在不同颜色空间中处理亮度通道或全部通道。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | 执行 CLAHE 的颜色空间 |
| `clip_limit` | `float` | `2.0` | 对比度裁剪阈值 |
| `tile_grid_size` | `tuple` | `(8, 8)` | 局部网格大小 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "clahe",
    "input.jpg",
    output="results/clahe/output.jpg",
    color_space="lab",
    clip_limit=2.0,
    tile_grid_size=(8, 8),
)
```

文件夹批处理：

```python
saved_paths = llv.predict(
    "clahe",
    "images/",
    output="results/clahe",
    progress_bar=True,
)
```

