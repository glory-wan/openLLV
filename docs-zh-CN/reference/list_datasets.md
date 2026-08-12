# openLLV.list_datasets()

`openLLV.list_datasets()` 返回每个已注册数据集查找名及别名。

## Function Form

```python
openLLV.list_datasets() -> List[str]
```

## Parameters

该函数没有参数。

## Returns

排序后的 `List[str]`，包含 `BaseDataset` 注册表接受的规范化小写类名键及别名键。

## Behavior Details

- 注册表查找不区分大小写。
- 结果包含查找键；`list_available()` 将其按数据集类去重为每类一条记录。

## Example

```python
import openLLV as llv

print(llv.list_datasets())
```

## Related

- 训练：[`openLLV.train()`](train.md)
- 所有组件类别：[`openLLV.list_available()`](list_available.md)
