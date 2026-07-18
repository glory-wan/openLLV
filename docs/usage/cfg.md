# Training Configuration

openLLV stores packaged YAML training configurations in `openLLV/deepLearning/config/`. The same package defines defaults and helpers for resolving, loading, and recursively merging configurations.

## Built-in YAML Files

| Config name | Model | Dataset | Loss |
| --- | --- | --- | --- |
| `CIDNet` | `CIDNet` | `CommonDataset` | `cidnet` |
| `DarkIR` | `DarkIR` | `CommonDataset` | `darkir` |
| `EnlightenGAN` | `EnlightenGAN` | `CommonDataset` | `enlightengan` |
| `KinD` | `KinD` | `CommonDataset` | `kind` |
| `KinD++` | `KinDPlusPlus` | `CommonDataset` | `kindplusplus` |
| `LEDNet` | `LEDNet` | `CommonDataset` | `lednet` |
| `LLFlow` | `LLFlow` | `CommonDataset` | `llflow` |
| `LLFormer` | `LLFormer` | `CommonDataset` | `llformer` |
| `LLNet` | `LLNet` | `CommonDataset` | `llnet` |
| `PairLIE` | `PairLIE` | `CommonDataset` | `pairlie` |
| `RetinexFormer` | `RetinexFormer` | `CommonDataset` | `retinexformer` |
| `RUAS` | `RUAS` | `CommonDataset` | `ruas` |
| `SCI` | `SCI` | `CommonDataset` | `sci` |
| `URetinexNet` | `URetinexNet` | `CommonDataset` | `uretinex` |
| `ZeroDCE` | `ZeroDCE` | `CommonDataset` | `zerodce` |
| `ZeroDCE++` | `ZeroDCEPlusPlus` | `CommonDataset` | `zerodce_extension` |
| `ZeroIG` | `ZeroIG` | `CommonDataset` | `zeroig` |

List packaged names programmatically:

```python
from openLLV.deepLearning.config import list_available_configs

print(list_available_configs())
```

Names may be a filename, filename stem, or the YAML `model.name`. Matching is case-insensitive and punctuation-insensitive.

## YAML Structure

```yaml
model:
  name: ZeroDCE
  params: {}

data:
  dataset: CommonDataset
  root_dir: path/to/your/dataset
  train_split: train
  val_split: _test
  batch_size: 4
  num_workers: 4
  pin_memory: true
  return_filename: true
  resize: null

loss:
  name: zerodce
  params: {}
  output_index: null
  output_key: null

optimizer:
  name: adam
  lr: 0.0001
  params:
    betas: [0.9, 0.999]
    weight_decay: 0.0

scheduler:
  name: cosineannealinglr
  params:
    T_max: 100
    eta_min: 0.000001

train:
  epochs: 100
  device: null
  output_dir: null
  save_every: 1
  validate_every: 1
  log_every: 10
  grad_clip: 1.0
  amp: false
  seed: 42
  resume: null
```

## Sections

### `model`

`name` selects a registered `LLVModel`. `params` is passed to that model's constructor.

### `data`

`dataset` selects a registered dataset. `root_dir` is required by the packaged `CommonDataset` configs. Shared constructor values belong in `params`; split-specific values belong in `train_params` and `val_params`. Explicit `train_input_dir`, `train_target_dir`, `val_input_dir`, and `val_target_dir` are also supported. Set `resize` to an integer for a square output or to `[height, width]` for an explicit size.

### `loss`

`name` selects a registered `BaseLoss`. `params` configures it. `output_key` or `output_index` can select a tensor from an unusual structured model output; normal `pred` dictionaries do not need either option.

### `optimizer`

`name` may select `adam`, `adamw`, `sgd`, or `rmsprop`. `lr` is the learning rate and `params` is forwarded to the optimizer.

### `scheduler`

Set `name: null` to disable scheduling. Otherwise `params` is forwarded to the selected PyTorch learning-rate scheduler.

### `train`

This section controls epochs, runtime device, output location, validation/checkpoint frequency, mixed precision, gradient clipping, reproducibility, and resume behavior. A null device resolves to CUDA when available, then MPS when available, and otherwise CPU.

## Default Values and Merge Helpers

`openLLV/deepLearning/config/__init__.py` exports:

- `DEFAULT_TRAIN_CONFIG`;
- `get_default_train_config()`, which returns an independent copy;
- `get_default_device()`;
- `deep_update()` for recursive dictionary merging;
- `get_config_path()` and `load_config()`;
- `list_available_configs()`.

Packaged YAML values are recursively merged over `DEFAULT_TRAIN_CONFIG`, then explicit trainer keyword overrides are applied.

## Override Examples

```python
import openLLV as llv

result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    batch_size=2,
    lr=5e-5,
    epochs=20,
    device="cpu",
)
```

Nested sections can be supplied directly:

```python
result = llv.train(
    {
        "model": {"name": "ZeroDCE"},
        "data": {
            "dataset": "CommonDataset",
            "root_dir": "datasets/my_dataset",
        },
        "loss": {"name": "zerodce"},
        "train": {"epochs": 20},
    }
)
```

Do not add `device` to `model.params`. Device placement is a trainer/predictor concern.
