# openLLV.list_losses()

`openLLV.list_losses()` returns every registered deep-learning loss lookup name and alias.

## Function Form

```python
openLLV.list_losses() -> List[str]
```

## Parameters

This function takes no parameters.

## Returns

A sorted `List[str]` of normalized lowercase keys in the `BaseLoss` registry, including the class name, declared canonical `name`, and aliases.

## Behavior Details

- Registry lookup is case-insensitive.
- The result is a lookup-key list, not a deduplicated class list; use `list_available()` for canonical class names plus alias arrays.

## Example

```python
import openLLV as llv

print(llv.list_losses())
```

## Related

- Training: [`openLLV.train()`](train.md)
- Deduplicated component metadata: [`openLLV.list_available()`](list_available.md)
