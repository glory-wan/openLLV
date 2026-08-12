# GCP

> 文档分组：传统低光图像增强

GCP 使用自适应 gamma、大气光估计、透射恢复和百分位范围调整实现 Gamma Correction Prior 增强。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://www.sciencedirect.com/science/article/abs/pii/S0031320323006994 |
| 官方源码 | https://github.com/TripleJ2543/Low_Light_Pattern_Recognition_2023 |
| 官方项目页 | 无 |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/LLIE/GCP.py` |
| 类名 | `GCP` |
| 注册名 | `GCP`（别名：`gcp`、`gcp-ms`；查找忽略大小写并去除首尾空白，因此 `GCP` 与 `gcp` 对应同一键） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

GCP 将灰度与 BGRA 输入转换为三通道工作图，并保留灰度/alpha 布局和源值域。每次调用先取得已存参数，再应用全部运行时关键字覆盖值，最后校验合并映射。因此未知运行时键会保留但后续不读取；已知覆盖值不会修改实例。工厂构造会过滤不支持的键并告警。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `gamma_max` | `float` | `6.0` | 最大自适应 gamma。 | 必须 `>= 1`；可运行时覆盖。 |
| `erosion_window` | `int` | `15` | 暗通道腐蚀核尺寸。 | 必须 `> 0`；可运行时覆盖。 |
| `atmospheric_bins` | `int` | `200` | 大气光估计直方图箱数。 | 必须 `> 0`；可运行时覆盖。 |
| `atmospheric_percentile` | `float` | `0.99` | 选择候选像素的暗通道比例。 | 必须在 `(0, 1)`；可运行时覆盖。 |
| `t_min` | `float` | `0.1` | 透射率下限。 | 必须在 `(0, 1]`；可运行时覆盖。 |
| `blur_ksize` | `int` | `7` | 高斯模糊核尺寸。 | 必须为正奇数；可运行时覆盖。 |
| `high_percentile` | `float` | `99.5` | 最终范围调整上百分位。 | 与 `low_percentile` 必须满足 `0 <= low < high <= 100`；可运行时覆盖。 |
| `low_percentile` | `float` | `0.5` | 最终范围调整下百分位。 | 与 `high_percentile` 必须满足 `0 <= low < high <= 100`；可运行时覆盖。 |
| `eps` | `float` | `0.000001` | 数值稳定下限。 | 必须 `> 0`；可运行时覆盖。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式。 | 其他值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 将输出转回输入 dtype。 | 非 `bool` 抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪到目标 dtype 有效范围。 | 非 `bool` 抛 `TypeError`。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("gcp-ms", "input.jpg", output="results/gcp/output.png", gamma_max=5.0, blur_ksize=5)
```

```python
from openLLV.tradition.algorithms.LLIE.GCP import GCP
enhancer = GCP(high_percentile=99.0)
result = enhancer("input.jpg", low_percentile=1.0)
```
