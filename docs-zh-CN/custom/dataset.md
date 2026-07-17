# 自定义数据集

数据集继承 `BaseDataset`，并在其模块导入时自动注册。基类负责图像读取、变换、按主干名称配对、数据划分元数据以及 PyTorch `Dataset` 行为。

## 1. 必需方法

大多数成对数据集只需实现 `_resolve_pair_dirs()`：

```python
from pathlib import Path
from typing import Optional, Tuple

from openLLV.data.datasets import BaseDataset


class MyDataset(BaseDataset):
    name = "my_dataset"
    aliases = ["mydata"]

    def _resolve_pair_dirs(
        self,
        low_dir: Optional[Path],
        high_dir: Optional[Path],
    ) -> Tuple[Path, Optional[Path]]:
        if low_dir is not None:
            return low_dir, high_dir

        split_root = self.root_dir / self.split
        return split_root / "input", split_root / "target"
```

如果文件无法按不区分大小写的主干名称配对，请覆盖 `_build_pairs()`，并返回由 `(input_path, target_path)` 组成的列表。

## 2. 返回样本

使用默认的 `return_filename=True` 时，每个数据项为：

```python
input_tensor, target_tensor, filename = dataset[index]
```

对于非成对数据集，`target_tensor` 可以为 `None`。默认的独立变换会把 PIL 图像转换为张量。

## 3. 注册

将数据集添加到 `openLLV/data/datasets/__init__.py`，使训练器能够导入它：

```python
from .MyDataset import MyDataset
```

然后进行验证：

```python
import openLLV as llv

print(llv.list_datasets())
```

## 4. 训练

```python
result = llv.train(
    "ZeroDCE",
    dataset="my_dataset",
    root_dir="datasets/my_dataset",
)
```

数据集专属的构造参数可以放在嵌套配置的 `data.params`、`data.train_params` 或 `data.val_params` 中。

## 5. 现有 CommonDataset

添加新类之前，请先确认 `CommonDataset` 是否已经满足需求。它支持显式的输入/目标目录，以及常见的 `train`、`test`、`val` 和 `validation` 数据划分结构。

