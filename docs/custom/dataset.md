# Custom Datasets

Datasets inherit from `BaseDataset` and register automatically when their module is imported. The base class handles image reading, transforms, stem-based pairing, split metadata, and PyTorch `Dataset` behavior.

## 1. Required Method

Most paired datasets only need to implement `_resolve_pair_dirs()`:

```python
from pathlib import Path
from typing import Optional, Tuple

from openLLV.data.datasets import BaseDataset


class MyDataset(BaseDataset):
    name = "my_dataset"
    aliases = ["mydata"]

    def _resolve_pair_dirs(
        self,
        input_dir: Optional[Path],
        target_dir: Optional[Path],
    ) -> Tuple[Path, Optional[Path]]:
        if input_dir is not None:
            return input_dir, target_dir

        split_root = self.root_dir / self.split
        return split_root / "input", split_root / "target"
```

If filenames cannot be paired by case-insensitive stem, override `_build_pairs()` and return a list of `(input_path, target_path)` pairs.

## 2. Returned Samples

With the default `return_filename=True`, each item is:

```python
input_tensor, target_tensor, filename = dataset[index]
```

`target_tensor` may be `None` for an unpaired dataset. The default image
transform optionally resizes each image and then converts it to a tensor. An
integer resize produces a square image; a two-item value is interpreted as
`(height, width)`.

## 3. Registration

Add the dataset to `openLLV/data/datasets/__init__.py` so the trainer imports it:

```python
from .MyDataset import MyDataset
```

Then verify:

```python
import openLLV as llv

print(llv.list_datasets())
```

## 4. Training

```python
result = llv.train(
    "ZeroDCE",
    dataset="my_dataset",
    root_dir="datasets/my_dataset",
    resize=(256, 384),
)
```

Dataset-specific constructor arguments can be placed under `data.params`, `data.train_params`, or `data.val_params` in a nested config.

## 5. Existing CommonDataset

Before adding a class, check whether `CommonDataset` is sufficient. It supports explicit input/target directories and common `train`, `test`, `val`, and `validation` split layouts.
