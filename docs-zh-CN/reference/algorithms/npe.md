# NPE

> 文档分组：传统低光图像增强

NPE 估计亮通照明，执行双对数映射，恢复反射细节，并向原图混合以保持自然度。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TIP.2013.2261309 |
| 官方源码 | 无 |
| 官方项目页 | 无 |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/LLIE/NPE.py` |
| 类名 | `NPE` |
| 注册名 | `NPE`（无别名；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

公共三通道输入和 NumPy 输出使用 RGB；NPE 仅把 BGR/BGRA 作为 `_enhance()` 内部布局，并在内部支持灰度图。它会保留灰度/alpha 布局。基类将解析后的源值域映射到 NPE 的 `[0,1]` 工作值域，随后恢复原输入的值域约定与可选 dtype。全部算法参数均可作为经校验的单次 `enhance()` 覆盖值，且不修改已存状态。工厂会忽略不支持的构造键，并在控制台说明其不会被使用且不影响计算；近似拼写会获得建议。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `sigma` | `float` | `15.0` | 亮通照明平滑的高斯尺度。 | 必须 `> 0`；可运行时覆盖。 |
| `illumination_floor` | `float` | `0.05` | 反射除法采用的最小照明。 | 必须在 `(0, 1]`；可运行时覆盖。 |
| `enhancement_strength` | `float` | `4.0` | 双对数照明映射强度。 | 必须 `> 0`；可运行时覆盖。 |
| `naturalness` | `float` | `0.35` | 原图混合权重。 | 必须在 `[0, 1]`；可运行时覆盖。 |
| `detail_weight` | `float` | `1.0` | 增强图减原图细节项的强度。 | 必须 `>= 0`；可运行时覆盖。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式。 | 其他值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 将输出转回输入 dtype。 | 非 `bool` 抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪到解析后的输入值域。 | 非 `bool` 抛 `TypeError`。 |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | 输入值域；自动区分通常的浮点 `[0,1]` 与 `[0,255]` 输入。 | 最大值 `<= 1` 的暗部字节值域浮点图须用 `"byte"`。自定义边界须有限且递增；图像值越界抛 `ValueError`。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("NPE", "input.jpg", output="results/npe/output.png", enhancement_strength=5.0, naturalness=0.25)
```

```python
from openLLV.tradition.algorithms.LLIE.NPE import NPE
enhancer = NPE(sigma=12.0)
result = enhancer("input.jpg", detail_weight=1.2)
```
