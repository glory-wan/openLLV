# openLLV.list_datasets()

`openLLV.list_datasets()` returns every registered dataset lookup name and alias.

## Function Form

```python
openLLV.list_datasets() -> List[str]
```

## Parameters

This function takes no parameters.

## Returns

A sorted `List[str]` of normalized lowercase `BaseDataset` registry keys, including class names and aliases.

## Behavior Details

- Registry lookup is case-insensitive.
- The result contains lookup keys; `list_available()` deduplicates them into one row per dataset class.

## Example

```python
import openLLV as llv

print(llv.list_datasets())
```

## Related

- Training: [`openLLV.train()`](train.md)
- All component categories: [`openLLV.list_available()`](list_available.md)
