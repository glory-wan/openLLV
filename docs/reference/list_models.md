# openLLV.list_models()

`openLLV.list_models()` returns every accepted registered deep-learning model lookup name, including aliases.

## Function Form

```python
openLLV.list_models() -> List[str]
```

## Parameters

This function takes no parameters.

## Returns

A sorted `List[str]` containing normalized registry keys for concrete `LLVModel` classes and their aliases. Keys are lowercase because registration applies `strip().lower()`. The model package is imported before this call.

## Behavior Details

- Model lookup is case-insensitive after surrounding whitespace is stripped.
- An implementation with aliases contributes multiple lookup strings; use `list_available()` for one deduplicated row per class.

## Example

```python
import openLLV as llv

if "zerodce" in llv.list_models():
    enhanced, path = llv.predict("ZeroDCE", "input.jpg")
```

## Related

- Deduplicated component metadata: [`openLLV.list_available()`](list_available.md)
- Traditional algorithms: [`openLLV.list_algorithms()`](list_algorithms.md)
