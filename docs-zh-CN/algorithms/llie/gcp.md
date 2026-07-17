# GCP

> 文档分组：低光图像增强（LLIE）

GCP 是 openLLV 实现的一种伽马校正先验低光增强算法，对应论文 **Low-light image enhancement using gamma correction prior in mixed color spaces**。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文链接 | https://www.sciencedirect.com/science/article/abs/pii/S0031320323006994 |
| 官方源代码 | https://github.com/TripleJ2543/Low_Light_Pattern_Recognition_2023 |
| 官方项目页面 | 无 |

## 算法介绍

GCP 并不是采用固定指数的普通伽马校正。它利用伽马校正先验构建像素级自适应伽马，并结合暗通道思想估计大气光和透射图，从而提高低光图像的亮度和可见细节。

openLLV 的实现参考了官方开源脚本。主要流水线包括：

1. 将图像转换到归一化浮点空间。
2. 对输入图像进行高斯平滑并反转，获得用于估计的低光退化表示。
3. 计算暗通道，并从暗通道响应较高的区域估计大气光。
4. 根据大气光对各通道进行归一化。
5. 根据每个像素的最大通道值生成自适应伽马，并估计透射图。
6. 根据透射图恢复增强图像，并使用百分位动态范围拉伸得到最终结果。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/LLIE/GCP.py` |
| 算法类名 | `GCP` |
| 注册名称 | `GCP` |
| 别名 | `gcp`、`gamma_correction_prior` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `gamma_max` | `float` | `6.0` | 像素自适应伽马的最大值 |
| `erosion_window` | `int` | `15` | 暗通道腐蚀核大小 |
| `atmospheric_bins` | `int` | `200` | 估计大气光时使用的直方图分箱数量 |
| `atmospheric_percentile` | `float` | `0.99` | 用于选择大气光候选区域的暗通道百分位比例 |
| `t_min` | `float` | `0.1` | 透射图下界 |
| `blur_ksize` | `int` | `7` | 高斯平滑核大小，必须为正奇数 |
| `high_percentile` | `float` | `99.5` | 最终动态范围拉伸的高百分位 |
| `low_percentile` | `float` | `0.5` | 最终动态范围拉伸的低百分位 |
| `eps` | `float` | `1e-6` | 防止除零的小数值 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "gcp",
    "input.jpg",
    output="results/gcp/output.jpg",
)
```

通过统一 Predictor 显式调用传统算法后端：

```python
from openLLV import Predictor

predictor = Predictor("gcp", backend="traditional")
predictor("images/", output="results/gcp")
```

传入自定义参数：

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "gcp",
    "input.jpg",
    output="results/gcp_custom.png",
    gamma_max=5.0,
    erosion_window=11,
    high_percentile=99.0,
    low_percentile=1.0,
)
```

