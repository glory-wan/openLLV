# openLLV.list_algorithms()

`openLLV.list_algorithms()` returns every accepted registered traditional-algorithm lookup name, including aliases.

## Function Form

```python
openLLV.list_algorithms() -> List[str]
```

## Parameters

This function takes no parameters.

## Returns

A sorted `List[str]` containing lowercase registry keys contributed by each concrete class name, an explicitly declared `name`, and declared aliases.

## Behavior Details

- Lookup is case-insensitive after surrounding whitespace is stripped.
- An implementation with aliases contributes multiple strings; use `list_available()` for one deduplicated row per class.

## Example

```python
import openLLV as llv

print(llv.list_algorithms())
enhanced, path = llv.predict("clahe", "input.jpg")
```

## Related

- Algorithm references: [`algorithms/`](algorithms/)
- Models: [`openLLV.list_models()`](list_models.md)
