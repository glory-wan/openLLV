# openLLV.evaluate()

`openLLV.evaluate()` computes registered image-quality metrics for a directory, saves JSON results, and returns either the result dictionary or its initialized `Evaluator`. `openLLV.eval()` is an alias.

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

`en_img_dir` is required, although the implementation uses an internal sentinel so the keyword-only `en` alias can supply it. `ref_img_dir` is optional; `ref` is its keyword-only alias.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `en_img_dir` | `Union[str, Path]` | required | Directory of enhanced images. Converted to `str` before `Evaluator` construction. | Required unless `en` is provided. |
| `ref_img_dir` | `Optional[Union[str, Path]]` | `None` | Optional reference-image directory. | Files are paired by `EvaluateDataset`; reference-required metrics produce `NaN` when no references are available. |
| `metrics` | `Optional[Union[str, List[str]]]` | `None` | Metric name or ordered list. `None` selects `['PSNR', 'SSIM']`; strings/list entries are uppercased. | Other container types raise `TypeError`; unknown or failed metrics are skipped with `UserWarning`. |
| `save_path` | `Optional[Union[str, Path]]` | `None` | JSON destination. | `None` writes `./results/eval.json`; parent directories are created. |
| `return_evaluator` | `bool` | `False` | Return the initialized `Evaluator` instead of its `.results` dictionary. | Evaluation and JSON saving still happen during construction. |
| `en` | `Union[str, Path]` | not supplied | Keyword-only backward-compatible alias for `en_img_dir`. | Supplying both names raises `TypeError`. |
| `ref` | `Optional[Union[str, Path]]` | not supplied | Keyword-only backward-compatible alias for `ref_img_dir`. | Supplying both names raises `TypeError`. |
| `device` | `Optional[Union[str, torch.device]]` | `None` | Metric device, forwarded through `**kwargs` to `Evaluator`; `None` selects CUDA when available, otherwise CPU. | Must be accepted by `torch.device`. |
| `batch_size` | `int` | `1` | Evaluation DataLoader batch size, forwarded through `**kwargs`. | Passed to `torch.utils.data.DataLoader`. |
| `num_workers` | `int` | `8` | Evaluation DataLoader worker count, forwarded through `**kwargs`. | Passed to `torch.utils.data.DataLoader`. |
| `data_range` | `float` | `1.0` | Shared metric constructor option forwarded through `**kwargs`; consumed by PSNR, SSIM, LPIPS, NIQE, and PI. | Represents the maximum image data range. |
| `window_size` | `int` | `11` | SSIM Gaussian-window size forwarded through `**kwargs`. | Used as convolution window size. |
| `sigma` | `float` | `1.5` | SSIM Gaussian-window standard deviation forwarded through `**kwargs`. | Used to build the Gaussian weights. |
| `net` | `str` | `"alex"` | LPIPS backbone forwarded through `**kwargs`. | Passed to `pyiqa.create_metric('lpips', ...)`; fallback omits it if unsupported. |
| `patch_size` | `int` | `50` | LOE approximation patch size forwarded through `**kwargs`. | Used as the spatial pooling divisor. |
| `scales` | `Optional[List[float]]` | `None` | MUSIQ caller metadata forwarded through `**kwargs`. | Stored by `MUSIQMetric`; the current compute path does not consume it. |

Arbitrary additional `**kwargs` are passed to every selected metric constructor and retained in the base metric's `config` when not consumed explicitly.

### Aliases

| Alias | Points to |
| --- | --- |
| `openLLV.eval()` | `openLLV.evaluate()` |
| `en` | `en_img_dir` |
| `ref` | `ref_img_dir` |

## Returns

With `return_evaluator=False`, returns:

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

With `return_evaluator=True`, returns the initialized `Evaluator`; its `results` property has the same structure. The saved JSON wraps results with `metadata` and uses `values` for the per-file metric mapping.

## Behavior Details

- `Evaluator(...)` performs evaluation immediately by calling `eval(...)` from its constructor.
- Registered metric matching is case-insensitive and accepts names with or without the `Metric` suffix.
- PSNR, SSIM, MSE, MAE, LPIPS, and LOE require references. NIQE, MUSIQ, and PI do not.
- Enhanced tensors are resized bilinearly to reference spatial dimensions before a metric runs; incompatible remaining shapes raise `ValueError` inside the metric.

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

This public method reruns the already selected metric instances on another directory and returns/saves a new result dictionary. Unlike the constructor, its `num_workers` default is `0`.

### Raises

| Exception | Condition |
| --- | --- |
| `TypeError` | Missing enhanced directory; both an argument and its alias are supplied; or `metrics` is neither `None`, `str`, nor `list`. |
| `ValueError` | Invalid device or metric input alignment can surface from `torch`/metric code. |
| `ImportError` | A selected pyiqa-backed metric (LPIPS, NIQE, MUSIQ, or PI) requires unavailable `pyiqa`; constructor failures are caught and emitted as warnings by `Evaluator`. |

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

- Available components: [`openLLV.list_available()`](list_available.md)
