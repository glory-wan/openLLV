# Training API

`openLLV.train()` creates a configuration-driven `Trainer` and runs the complete training loop. The trainer owns runtime device placement and can instantiate every concrete model derived from `LLVModel`.

## Function Form

```python
openLLV.train(config=None, **kwargs)
```

`config` may be:

- a packaged configuration name such as `"ZeroDCE"` or `"ZeroDCE.yaml"`;
- a YAML file path;
- a nested configuration dictionary;
- `None`, when all required values are passed as keyword arguments.

Keyword arguments override values loaded from the selected configuration.

## Dataset Layout

All packaged configs currently use `CommonDataset`. Its preferred paired layout is:

```text
dataset_root/
  train/
    input/
      image_001.png
    target/
      image_001.png
  val/
    input/
      image_001.png
    target/
      image_001.png
```

Filenames are paired by case-insensitive stem. The default validation split is `_test`, which resolves common `test`, `val`, and `validation` directory names. Explicit `train_low_dir`, `train_high_dir`, `val_low_dir`, and `val_high_dir` overrides are also supported.

## Train with a Built-in Config

```python
import openLLV as llv

result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=4,
    device="cuda",
)
```

Built-in names are case-insensitive and punctuation-insensitive. For example, `"ZeroDCEPlusPlus"` resolves `ZeroDCE++.yaml`.

## Train with a YAML File

```python
result = llv.train(
    "configs/experiment.yaml",
    lr=5e-5,
    amp=True,
)
```

## Train with a Dictionary

```python
config = {
    "model": {"name": "ZeroDCE", "params": {}},
    "data": {
        "dataset": "CommonDataset",
        "root_dir": "datasets/my_dataset",
        "batch_size": 4,
    },
    "loss": {"name": "zerodce", "params": {}},
    "optimizer": {"name": "adam", "lr": 1e-4},
    "train": {"epochs": 10, "device": "cuda"},
}

result = llv.train(config)
```

## Direct Trainer Use

```python
from openLLV.deepLearning import Trainer

trainer = Trainer("ZeroDCE", root_dir="datasets/my_dataset")
result = trainer.train()
```

The returned dictionary contains the history, best validation loss, and checkpoint-directory path. Training timestamps are stored in the checkpoint and saved training configuration.

## Common Overrides

| Keyword | Configuration target |
| --- | --- |
| `model`, `model_name` | `model.name` |
| `model_params` | `model.params` |
| `dataset`, `root_dir`, `batch_size`, `num_workers` | `data.*` |
| `loss`, `loss_params` | `loss.*` |
| `optimizer`, `lr`, `optimizer_params` | `optimizer.*` |
| `scheduler`, `scheduler_params` | `scheduler.*` |
| `epochs`, `device`, `amp`, `grad_clip` | `train.*` |
| `output_dir`, `resume`, `save_every` | `train.*` |

Unknown flat keywords raise `TypeError` instead of being silently ignored.

## Outputs and Resume

By default, training writes `checkpoints/<Model>_<Dataset>/` containing:

```text
checkpoints/<Model>_<Dataset>/
  checkpoints/
    best.pt
    last.pt
  logs/
    history.json
  <Model>.yaml
```

Resume from an openLLV checkpoint with:

```python
result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    resume="checkpoints/ZeroDCE_CommonDataset/checkpoints/last.pt",
)
```

Use `strict_resume=False` only when intentionally loading a partially compatible state dictionary.
