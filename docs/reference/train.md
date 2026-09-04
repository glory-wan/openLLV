# openLLV.train()

`openLLV.train()` constructs `Trainer`, runs its complete training loop, and returns training history plus checkpoint information.

## Function Form

```python
openLLV.train(config=None, **kwargs)
```

`Trainer(config=None, **kwargs)` accepts the same construction arguments. Its public `train()` method takes no arguments and returns the same result as `openLLV.train()`.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Union[str, Path, Dict[str, Any]]]` | `None` | Built-in YAML name, YAML path, nested configuration dictionary, or no base configuration. | Non-mapping YAML raises `ValueError`; missing path/name raises `FileNotFoundError`. Built-in matching ignores case and punctuation. |
| `model`, `model_name` | `str`, `Path`, `LLVModel`, or `Type[LLVModel]` | `None` | Alias flat keys for `model.name`. A path loads an openLLV checkpoint. | Required; classes must inherit `LLVModel`. |
| `model_params` | `Dict[str, Any]` | `{}` | Maps to `model.params`; `mode="train"` is added when the model is built. | Must be a dictionary. |
| `model_config` | `Dict[str, Any]` | `{}` | Merges directly into the nested `model` section. | Must be a dictionary. |
| `dataset`, `dataset_name` | `str`, `BaseDataset`, or `Type[BaseDataset]` | `"CommonDataset"` | Alias flat keys for `data.dataset`. | Registered name lookup is case-insensitive. |
| `root_dir` | `Optional[Union[str, Path]]` | `None` | Maps to `data.root_dir`. | Required unless `dataset` is an existing `BaseDataset` instance. |
| `batch_size` | `int` | `4` | Maps to `data.batch_size`. | Positive non-boolean integer. |
| `num_workers` | `int` | `0` | Maps to `data.num_workers`. | Non-negative non-boolean integer. |
| `pin_memory` | `bool` | `True` | Maps to `data.pin_memory`. | Passed to DataLoader. |
| `shuffle` | `bool` | `True` | Maps to `data.shuffle` for the training loader. | — |
| `drop_last` | `bool` | `False` | Maps to `data.drop_last`. | — |
| `train_split` | `str` | `"train"` | Maps to `data.train_split`. | Dataset-specific. |
| `val_split` | `str` | `"val"` | Maps to `data.val_split`. | Dataset-specific. |
| `return_filename` | `bool` | `True` | Maps to `data.return_filename`. | — |
| `resize` | `Optional[Union[int, Sequence[int]]]` | `None` | Maps to `data.resize`; an integer means a square, a sequence supplies height/width. | Dataset validation applies. |
| `train_input_dir` | `Optional[str]` | `None` | Maps to `data.train_input_dir`. | Overrides dataset-specific training input directory. |
| `train_target_dir` | `Optional[str]` | `None` | Maps to `data.train_target_dir`. | May be omitted for reference-free training. |
| `val_input_dir` | `Optional[str]` | `None` | Maps to `data.val_input_dir`. | — |
| `val_target_dir` | `Optional[str]` | `None` | Maps to `data.val_target_dir`. | — |
| `data_params` | `Dict[str, Any]` | `{}` | Maps to shared `data.params`. | Must be a dictionary. |
| `train_params` | `Dict[str, Any]` | `{}` | Maps to `data.train_params`. | Must be a dictionary. |
| `val_params` | `Dict[str, Any]` | `{}` | Maps to `data.val_params`. | Must be a dictionary. |
| `data` | `Dict[str, Any]` | default `data` section | Merges a complete nested data section. | Must be a dictionary. |
| `loss`, `loss_name` | `Optional[Union[str, nn.Module]]` | `None` | Alias flat keys for `loss.name`. The model's default loss may be used when omitted. | Registered string, module, or model-supported default. |
| `loss_params` | `Dict[str, Any]` | `{}` | Maps to `loss.params`. | Passed to the loss constructor. |
| `output_index` | `Optional[int]` | `None` | Maps to `loss.output_index` for tuple/list model outputs. | Must select an existing output. |
| `output_key` | `Optional[str]` | `None` | Maps to `loss.output_key` for mapping model outputs. | Must select an existing key. |
| `optimizer`, `optimizer_name` | `str` | `"adam"` | Alias flat keys for `optimizer.name`. | Supported names are resolved by Trainer. |
| `lr` | `float` | `1e-4` | Maps to `optimizer.lr`. | Must be greater than `0`; booleans are rejected. |
| `optimizer_params` | `Dict[str, Any]` | `{}` | Maps to `optimizer.params`. | Passed to the optimizer constructor. |
| `scheduler`, `scheduler_name` | `Optional[str]` | `None` | Alias flat keys for `scheduler.name`; `None` disables scheduling. | Supported names are resolved by Trainer. |
| `scheduler_params` | `Dict[str, Any]` | `{}` | Maps to `scheduler.params`. | Passed to the scheduler constructor. |
| `epochs` | `int` | `100` | Maps to `train.epochs`. | Positive non-boolean integer. |
| `output_dir` | `Optional[Union[str, Path]]` | `None` | Maps to `train.output_dir`. | `None` resolves to `checkpoints/<Model>_<Dataset>`. |
| `save_every` | `int` | `0` | Maps to `train.save_every`; values above zero save an additional `epoch_<epoch>.pt` at that epoch interval. | Non-negative non-boolean integer; `0` disables numbered snapshots. |
| `validate_every` | `int` | `1` | Maps to `train.validate_every`. | Positive non-boolean integer. |
| `log_every` | `int` | `10` | Maps to `train.log_every`. | Positive non-boolean integer. |
| `grad_clip` | `Optional[float]` | `None` | Maps to `train.grad_clip`. | When set, must be greater than `0`. |
| `amp` | `bool` | `False` | Maps to `train.amp`; effective only on CUDA. | — |
| `resume`, `resume_path` | `Optional[Union[str, Path]]` | `None` | Alias flat keys for `train.resume`. | Checkpoint must be loadable by Trainer. |
| `strict_resume` | `bool` | `True` | Maps to `train.strict_resume`. | Controls state-dictionary strictness. |
| `seed` | `Optional[int]` | `42` | Maps to `train.seed`; `None` disables seed setup. | Integer or `None`; booleans are rejected. |
| `device` | `Union[str, torch.device]` | best available device | Maps to `train.device`; default preference is CUDA, then MPS, then CPU. | Requesting unavailable CUDA/MPS raises `RuntimeError`. |
| `device_ids` | `Optional[Union[List[int], Tuple[int, ...]]]` | `None` | Maps to `train.device_ids`; two or more CUDA indices enable single-process `DataParallel`. | Must be non-empty, unique, non-negative, available CUDA indices; the first entry must match an explicitly indexed `device`. |
| `progress_bar` | `bool` | `True` | Maps to `train.progress_bar`. | Controls tqdm training/validation bars. |
| `model`, `loss`, `optimizer`, `scheduler`, `train` as dictionaries | `Dict[str, Any]` | corresponding default section | A dictionary under one of these names merges directly into that nested section instead of acting as its flat alias. | Section values must be dictionaries after merging. |

Any unknown flat `**kwargs` key raises `TypeError`; it is never silently ignored.

## Returns

```python
{
    "history": [
        {
            "epoch": int,
            "train_loss": float,
            "val_loss": Optional[float],
            "lr": float,
            "seconds": float,
        },
        ...,
    ],
    "best_val_loss": float,
    "checkpoint_dir": str,
}
```

If validation never produces a loss, `best_val_loss` remains positive infinity.

## Behavior Details

- Configuration precedence is defaults, then `config`, then flat/nested `**kwargs` overrides.
- Trainer owns device placement. It builds the model, datasets, loss, optimizer, scheduler, AMP scaler, output folders, and optional resume state before `train()` starts.
- With two or more `device_ids`, Trainer uses single-machine, single-process `torch.nn.DataParallel`. The original model remains the optimizer/checkpoint owner, so saved state-dictionary keys do not gain a `module.` prefix.
- `last.pt` and `best.pt` are overwritten after every epoch, independently of `save_every`. If `save_every > 0`, an additional `epoch_<epoch>.pt` is saved whenever the epoch number is divisible by `save_every`.
- Model config names such as `ZeroDCEPlusPlus` can resolve punctuation-bearing YAML files such as `ZeroDCE++.yaml`.

### Raises

| Exception | Condition |
| --- | --- |
| `TypeError` | Invalid config/section type, unknown flat key, invalid model class/input, invalid seed/device-ID type, or incompatible training objects. |
| `ValueError` | Missing model/dataset root, invalid positive/non-negative settings, non-positive learning rate/gradient clip, invalid device text, or invalid/unavailable CUDA device IDs. |
| `FileNotFoundError` | YAML/checkpoint path or built-in configuration cannot be resolved. |
| `RuntimeError` | Explicit CUDA/MPS device is unavailable, or `device_ids` is used with a non-CUDA device. |
| `FloatingPointError` | A computed loss is NaN or infinite. |

## Examples

```python
import openLLV as llv

result = llv.train(
    "ZeroDCE",
    root_dir="data/LOL-v1",
    epochs=20,
    batch_size=8,
    lr=1e-4,
    output_dir="runs/zero_dce",
)
```

```python
trainer = llv.Trainer(
    model="ZeroDCE",
    dataset="CommonDataset",
    root_dir="data/LOL-v1",
    scheduler=None,
    amp=True,
)
result = trainer.train()
```

```python
result = llv.train(
    "ZeroDCE",
    root_dir="data/LOL-v1",
    batch_size=8,
    device="cuda",
    device_ids=[0, 1],
)
```

## Related

- Model component references: [`models/`](models/)
- Prediction: [`openLLV.predict()`](predict.md)
