# openLLV.list_metrics()

`openLLV.list_metrics()` returns the simple names of all registered image-quality metrics.

## Function Form

```python
openLLV.list_metrics() -> List[str]
```

## Parameters

This function takes no parameters.

## Returns

A `List[str]` in metric registration/insertion order (not explicitly sorted). The `Metric` class suffix is removed, so `PSNRMetric` is returned as `"PSNR"`.

## Behavior Details

- Metric implementations are imported before registry lookup.
- `evaluate()` matches names case-insensitively and also accepts the `Metric` suffix.

## Example

```python
import openLLV as llv

metrics = llv.list_metrics()
results = llv.evaluate("enhanced", "reference", metrics=metrics[:2])
```

## Related

- Evaluation: [`openLLV.evaluate()`](evaluate.md)
- All component categories: [`openLLV.list_available()`](list_available.md)
