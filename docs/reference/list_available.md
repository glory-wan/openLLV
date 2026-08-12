# openLLV.list_available()

`openLLV.list_available()` returns deduplicated metadata for every registered public component class, grouped by category.

## Function Form

```python
openLLV.list_available() -> Dict[str, List[Dict[str, Any]]]
```

## Parameters

This function takes no parameters.

## Returns

```python
{
    "models": [{"name": str, "aliases": List[str]}, ...],
    "algorithms": [{"name": str, "aliases": List[str]}, ...],
    "metrics": [{"name": str, "aliases": List[str]}, ...],
    "losses": [{"name": str, "aliases": List[str]}, ...],
    "datasets": [{"name": str, "aliases": List[str]}, ...],
}
```

Each implementation class appears once per category. `name` is the Python class name; `aliases` is copied from the class's `aliases` attribute (a single string is normalized to a one-element list).

## Behavior Details

- Classes within each category are sorted case-insensitively by class name.
- This function differs from `list_models()`, `list_algorithms()`, `list_losses()`, and `list_datasets()`: those return every accepted lookup key, whereas this function deduplicates registry values by implementation class.
- Algorithm `name` is also a class name here, not necessarily its `LLVEnhancer.name` registry label. For example, the DCP implementation row uses class name `DarkChannel` and alias `dcp`. Consequently, a row is display metadata and is not a complete list of accepted lookup keys; use the corresponding flat `list_*()` function for that.

## Example

```python
import openLLV as llv

available = llv.list_available()
for model in available["models"]:
    print(model["name"], model["aliases"])
```

## Related

- Models: [`openLLV.list_models()`](list_models.md)
- Algorithms: [`openLLV.list_algorithms()`](list_algorithms.md)
- Metrics: [`openLLV.list_metrics()`](list_metrics.md)
- Losses: [`openLLV.list_losses()`](list_losses.md)
- Datasets: [`openLLV.list_datasets()`](list_datasets.md)
