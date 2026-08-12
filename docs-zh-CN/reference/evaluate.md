# openLLV.evaluate()

`openLLV.evaluate()` 对目录计算已注册的图像质量指标、保存 JSON 结果，并返回结果字典或初始化后的 `Evaluator`。`openLLV.eval()` 是其别名。

## Function Form

```python
openLLV.evaluate(
    en_img_dir,
    ref_img_dir=None,
    metrics=None,
    save_path=None,
    return_evaluator=False,
    *,
    en,
    ref,
    **kwargs,
)
```

`en_img_dir` 为必填，但实现使用内部哨兵值，使仅关键字别名 `en` 可以代替它。`ref_img_dir` 可选；`ref` 是其仅关键字别名。

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `en_img_dir` | `Union[str, Path]` | 必填 | 增强图目录；构造 `Evaluator` 前转为 `str`。 | 除非提供 `en`，否则必填。 |
| `ref_img_dir` | `Optional[Union[str, Path]]` | `None` | 可选参考图目录。 | 文件由 `EvaluateDataset` 配对；没有参考图时，需要参考图的指标产生 `NaN`。 |
| `metrics` | `Optional[Union[str, List[str]]]` | `None` | 指标名或有序列表。`None` 选择 `['PSNR', 'SSIM']`；字符串/列表项转大写。 | 其它容器类型抛 `TypeError`；未知或创建失败的指标以 `UserWarning` 跳过。 |
| `save_path` | `Optional[Union[str, Path]]` | `None` | JSON 目标位置。 | `None` 写入 `./results/eval.json`；自动创建父目录。 |
| `return_evaluator` | `bool` | `False` | 返回初始化后的 `Evaluator`，而不是其 `.results` 字典。 | 构造期间仍会执行评估并保存 JSON。 |
| `en` | `Union[str, Path]` | 未提供 | `en_img_dir` 的仅关键字向后兼容别名。 | 两个名称同时提供抛 `TypeError`。 |
| `ref` | `Optional[Union[str, Path]]` | 未提供 | `ref_img_dir` 的仅关键字向后兼容别名。 | 两个名称同时提供抛 `TypeError`。 |
| `device` | `Optional[Union[str, torch.device]]` | `None` | 经 `**kwargs` 转发给 `Evaluator` 的指标设备；`None` 为 CUDA（若可用），否则 CPU。 | 必须可被 `torch.device` 接受。 |
| `batch_size` | `int` | `1` | 经 `**kwargs` 转发的评估 DataLoader batch size。 | 传给 `torch.utils.data.DataLoader`。 |
| `num_workers` | `int` | `8` | 经 `**kwargs` 转发的评估 DataLoader worker 数。 | 传给 `torch.utils.data.DataLoader`。 |
| `data_range` | `float` | `1.0` | 经 `**kwargs` 转发的共享指标构造选项；PSNR、SSIM、LPIPS、NIQE、PI 消费。 | 表示图像数据范围最大值。 |
| `window_size` | `int` | `11` | 经 `**kwargs` 转发的 SSIM 高斯窗口大小。 | 作为卷积窗口大小。 |
| `sigma` | `float` | `1.5` | 经 `**kwargs` 转发的 SSIM 高斯窗口标准差。 | 用于构造高斯权重。 |
| `net` | `str` | `"alex"` | 经 `**kwargs` 转发的 LPIPS backbone。 | 传给 `pyiqa.create_metric('lpips', ...)`；不支持时回退为省略该参数。 |
| `patch_size` | `int` | `50` | 经 `**kwargs` 转发的 LOE 近似 patch 大小。 | 作为空间池化除数。 |
| `scales` | `Optional[List[float]]` | `None` | 经 `**kwargs` 转发的 MUSIQ 调用方元数据。 | 由 `MUSIQMetric` 保存；当前计算路径不消费它。 |

任意额外 `**kwargs` 都传给每个所选指标的构造器；未被显式消费的键保留在基类指标的 `config` 中。

### Aliases

| Alias | Points to |
| --- | --- |
| `openLLV.eval()` | `openLLV.evaluate()` |
| `en` | `en_img_dir` |
| `ref` | `ref_img_dir` |

## Returns

`return_evaluator=False` 时返回：

```python
{
    "filenames": [str, ...],
    "metrics": {metric: {filename: float}},
    "statistics": {
        metric: {
            "mean": float,
            "std": float,
            "min": float,
            "max": float,
            "valid_count": int,
            "total_count": int,
            "better": "↑" or "↓",
        }
    },
}
```

`return_evaluator=True` 时返回初始化后的 `Evaluator`；其 `results` 属性结构相同。保存的 JSON 使用 `metadata` 包装结果，并以 `values` 保存逐文件指标映射。

## Behavior Details

- `Evaluator(...)` 在构造器中调用 `eval(...)`，因此初始化时立即执行评估。
- 已注册指标匹配不区分大小写，接受带或不带 `Metric` 后缀的名称。
- PSNR、SSIM、MSE、MAE、LPIPS、LOE 需要参考图；NIQE、MUSIQ、PI 不需要。
- 指标运行前会用双线性插值把增强 tensor 调整到参考图空间尺寸；仍不兼容的形状在指标内部抛 `ValueError`。

### `Evaluator.eval()`

```python
evaluator.eval(
    en_img_dir,
    ref_img_dir=None,
    save_path=None,
    batch_size=1,
    num_workers=0,
)
```

该公开方法使用已经选择的指标实例重新评估另一目录，并返回/保存新结果字典。与构造器不同，它的 `num_workers` 默认值是 `0`。

### Raises

| Exception | Condition |
| --- | --- |
| `TypeError` | 缺少增强图目录；参数及其别名同时提供；或 `metrics` 不是 `None`、`str`、`list`。 |
| `ValueError` | 无效设备或指标输入对齐错误可能从 `torch`/指标代码抛出。 |
| `ImportError` | 所选 pyiqa 指标（LPIPS、NIQE、MUSIQ、PI）需要未安装的 `pyiqa`；`Evaluator` 会捕获构造失败并以警告报告。 |

## Examples

```python
import openLLV as llv

results = llv.evaluate(
    "results/enhanced",
    "data/reference",
    metrics=["PSNR", "SSIM", "LPIPS"],
    save_path="results/metrics.json",
    batch_size=4,
)
```

```python
evaluator = llv.evaluate(
    en="results/enhanced",
    metrics="NIQE",
    return_evaluator=True,
    num_workers=0,
)
print(evaluator.results["statistics"])
```

## Related

- 可用组件：[`openLLV.list_available()`](list_available.md)
