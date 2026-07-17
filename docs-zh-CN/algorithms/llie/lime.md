# LIME

> 文档分组：低光图像增强（LLIE）

LIME 是一种基于照明图估计的传统低光图像增强算法。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TIP.2016.2639450 |
| 论文名称 | Low-light Image Enhancement via Illumination Map Estimation |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

LIME 根据输入图像的最大颜色通道估计照明图。随后使用边缘保持平滑细化照明图，并利用该照明图恢复更明亮的、类似反射率的图像。

在 openLLV 中，照明图通过引导滤波进行细化。增强图像通过输入图像除以经过伽马调整的照明图得到。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/LLIE/LIME.py` |
| 算法类名 | `LIME` |
| 注册名称 | `LIME` |
| 别名 | `lime`、`illumination_map_estimation` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `gamma` | `float` | `0.8` | 应用于细化照明图的伽马值 |
| `guided_radius` | `int` | `15` | 引导滤波半径 |
| `guided_eps` | `float` | `1e-3` | 引导滤波正则项 |
| `illumination_floor` | `float` | `0.05` | 照明下界 |
| `exposure` | `float` | `1.0` | 全局曝光乘数 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "lime",
    "input.jpg",
    output="results/lime/output.jpg",
    gamma=0.8,
)
```

文件夹批处理：

```python
saved_paths = llv.predict(
    "lime",
    "images/",
    output="results/lime",
    progress_bar=True,
)
```

