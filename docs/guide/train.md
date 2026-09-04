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

Filenames are paired by case-insensitive stem. The generic Trainer default validation split is `"val"`; packaged configs may override it, commonly with `_test`, which resolves common `test`, `val`, and `validation` directory names. Explicit `train_input_dir`, `train_target_dir`, `val_input_dir`, and `val_target_dir` overrides are also supported.

## Train with a Built-in Config

```python
import openLLV as llv

result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=4,
    resize=512,
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
        "resize": [384, 512],
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

## Multi-GPU Training

Pass two or more CUDA indices through `device_ids` to enable single-process data-parallel training:

```python
result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    batch_size=8,
    device="cuda",
    device_ids=[0, 1],
)
```

The first index is the primary device for the model, loss, and gathered outputs; each batch is split across the listed GPUs by `torch.nn.DataParallel`. If `device` includes an explicit index, it must match the first `device_ids` entry. `None` keeps the existing single-device behavior, while a one-item list selects that CUDA device without enabling parallelism. This is single-machine, single-process parallelism, not distributed or multi-node training.

## Trainer Parameters

The defaults below are the generic Trainer defaults before a built-in YAML, custom YAML, or configuration dictionary overrides them. Component-specific keys inside `model_params`, `loss_params`, `optimizer_params`, `scheduler_params`, and dataset parameter dictionaries are intentionally not expanded.

| Parameter | Default | Description |
| --- | --- | --- |
| `config` | `None` | Base configuration: a built-in config name, YAML path, nested dictionary, or `None`. |
| `model` | `None` | Sets `model.name`; accepts a registered name, checkpoint path, `LLVModel` class/instance, or a complete `model` section dictionary. A model is required. |
| `model_name` | `None` | Alias of `model` when selecting `model.name`. |
| `model_params` | `{}` | Dictionary forwarded to the selected model as `model.params`; model-specific keys are not listed here. |
| `model_config` | `{}` | Dictionary merged directly into the complete `model` section. |
| `dataset` | `"CommonDataset"` | Selects the registered dataset, dataset class, or existing dataset instance. |
| `dataset_name` | `"CommonDataset"` | Alias of `dataset`. |
| `root_dir` | `None` | Dataset root directory; required unless `dataset` is an existing dataset instance. |
| `batch_size` | `4` | Number of samples per training or validation batch; must be a positive integer. |
| `num_workers` | `0` | Number of DataLoader worker processes; must be a non-negative integer. |
| `pin_memory` | `True` | Enables DataLoader pinned memory when training on CUDA. |
| `shuffle` | `True` | Shuffles the training DataLoader. Validation is never shuffled. |
| `drop_last` | `False` | Drops the final incomplete training batch. |
| `train_split` | `"train"` | Dataset split name used for training. |
| `val_split` | `"val"` | Dataset split name used for validation; a false-like value disables automatic validation-dataset construction. |
| `return_filename` | `True` | Requests filenames from datasets that support this option. |
| `resize` | `None` | Dataset output size; `None` preserves size, an integer creates a square, and a two-item sequence means `(height, width)`. |
| `train_input_dir` | `None` | Explicit training-input directory overriding dataset layout discovery. |
| `train_target_dir` | `None` | Explicit training-target directory; may remain `None` for reference-free training. |
| `val_input_dir` | `None` | Explicit validation-input directory overriding dataset layout discovery. |
| `val_target_dir` | `None` | Explicit validation-target directory. |
| `data_params` | `{}` | Shared keyword dictionary passed to training and validation dataset constructors; dataset-specific keys are not listed here. |
| `train_params` | `{}` | Training-split dataset constructor overrides merged over `data_params`. |
| `val_params` | `{}` | Validation-split dataset constructor overrides merged over `data_params`. |
| `data` | Default `data` section | Complete nested data-section dictionary merged over the generic data defaults. |
| `loss` | `None` | Sets `loss.name`; accepts a registered/importable loss, loss class/instance, or complete `loss` section dictionary. When omitted, Trainer infers a matching loss or uses Charbonnier loss. |
| `loss_name` | `None` | Alias of `loss` when selecting `loss.name`. |
| `loss_params` | `{}` | Dictionary passed to the selected loss constructor; loss-specific keys are not listed here. |
| `output_index` | `None` | Selects one element from a tuple/list model output for loss computation. |
| `output_key` | `None` | Selects one value from a dictionary model output for loss computation. |
| `optimizer` | `"adam"` | Selects the optimizer name/class/instance or supplies a complete `optimizer` section dictionary. Built-in names are Adam, AdamW, SGD, and RMSprop. |
| `optimizer_name` | `"adam"` | Alias of `optimizer` when selecting `optimizer.name`. |
| `lr` | `1e-4` | Optimizer learning rate; must be greater than zero. |
| `optimizer_params` | `{}` | Additional optimizer constructor parameters; optimizer-specific keys are not listed here. |
| `scheduler` | `None` | Selects the scheduler name/instance or supplies a complete `scheduler` section dictionary; `None` disables scheduling. |
| `scheduler_name` | `None` | Alias of `scheduler` when selecting `scheduler.name`. |
| `scheduler_params` | `{}` | Scheduler constructor parameters; scheduler-specific keys are not listed here. |
| `epochs` | `100` | Total number of training epochs; must be a positive integer. |
| `output_dir` | `None` | Run-output directory; `None` resolves to `checkpoints/<Model>_<Dataset>`. |
| `save_every` | `1` | Saves `last.pt` every this many epochs; must be a positive integer. |
| `validate_every` | `1` | Runs validation every this many epochs when a validation loader exists; must be a positive integer. |
| `log_every` | `10` | Updates the training progress display every this many batches; must be a positive integer. |
| `grad_clip` | `None` | Maximum gradient norm; `None` disables clipping, otherwise the value must be greater than zero. |
| `amp` | `False` | Enables automatic mixed precision when the resolved device is CUDA. |
| `resume` | `None` | openLLV training-checkpoint path used to restore model and training state. |
| `resume_path` | `None` | Alias of `resume`. |
| `strict_resume` | `True` | Controls strict model state-dictionary loading during resume. |
| `seed` | `42` | Python, NumPy, and PyTorch random seed; `None` disables seed setup. |
| `device` | CUDA, then MPS, then CPU | Training device selected from the best available backend unless explicitly provided. |
| `device_ids` | `None` | Ordered CUDA device indices for single-process data-parallel training; two or more entries enable `DataParallel`, and the first entry is the primary device. |
| `progress_bar` | `True` | Enables tqdm progress bars for training and validation. |
| `train` | Default `train` section | Complete nested training-section dictionary merged over the generic training defaults. |

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
