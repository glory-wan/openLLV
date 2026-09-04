# openLLV.train()

`openLLV.train()` 构造 `Trainer`、执行完整训练循环，并返回训练历史及 checkpoint 信息。

## Function Form

```python
openLLV.train(config=None, **kwargs)
```

`Trainer(config=None, **kwargs)` 接受相同构造参数；其公开 `train()` 方法无参数，返回值与 `openLLV.train()` 相同。

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Union[str, Path, Dict[str, Any]]]` | `None` | 内置 YAML 名、YAML 路径、嵌套配置字典或无基础配置。 | 非映射 YAML 抛 `ValueError`；路径/名称不存在抛 `FileNotFoundError`；内置名匹配忽略大小写与标点。 |
| `model`, `model_name` | `str`、`Path`、`LLVModel` 或 `Type[LLVModel]` | `None` | `model.name` 的平铺别名；路径加载 openLLV checkpoint。 | 必填；类必须继承 `LLVModel`。 |
| `model_params` | `Dict[str, Any]` | `{}` | 映射到 `model.params`；构建模型时加入 `mode="train"`。 | 必须为字典。 |
| `model_config` | `Dict[str, Any]` | `{}` | 直接合并进嵌套 `model` 节。 | 必须为字典。 |
| `dataset`, `dataset_name` | `str`、`BaseDataset` 或 `Type[BaseDataset]` | `"CommonDataset"` | `data.dataset` 的平铺别名。 | 注册名匹配不区分大小写。 |
| `root_dir` | `Optional[Union[str, Path]]` | `None` | 映射到 `data.root_dir`。 | 除非 `dataset` 是已有 `BaseDataset` 实例，否则必填。 |
| `batch_size` | `int` | `4` | 映射到 `data.batch_size`。 | 正的非布尔整数。 |
| `num_workers` | `int` | `0` | 映射到 `data.num_workers`。 | 非负、非布尔整数。 |
| `pin_memory` | `bool` | `True` | 映射到 `data.pin_memory`。 | 传给 DataLoader。 |
| `shuffle` | `bool` | `True` | 映射到训练 loader 的 `data.shuffle`。 | — |
| `drop_last` | `bool` | `False` | 映射到 `data.drop_last`。 | — |
| `train_split` | `str` | `"train"` | 映射到 `data.train_split`。 | 依赖数据集。 |
| `val_split` | `str` | `"val"` | 映射到 `data.val_split`。 | 依赖数据集。 |
| `return_filename` | `bool` | `True` | 映射到 `data.return_filename`。 | — |
| `resize` | `Optional[Union[int, Sequence[int]]]` | `None` | 映射到 `data.resize`；整数表示正方形，序列给出高/宽。 | 由数据集校验。 |
| `train_input_dir` | `Optional[str]` | `None` | 映射到 `data.train_input_dir`。 | 覆盖数据集训练输入目录。 |
| `train_target_dir` | `Optional[str]` | `None` | 映射到 `data.train_target_dir`。 | 无参考训练可省略。 |
| `val_input_dir` | `Optional[str]` | `None` | 映射到 `data.val_input_dir`。 | — |
| `val_target_dir` | `Optional[str]` | `None` | 映射到 `data.val_target_dir`。 | — |
| `data_params` | `Dict[str, Any]` | `{}` | 映射到共享 `data.params`。 | 必须为字典。 |
| `train_params` | `Dict[str, Any]` | `{}` | 映射到 `data.train_params`。 | 必须为字典。 |
| `val_params` | `Dict[str, Any]` | `{}` | 映射到 `data.val_params`。 | 必须为字典。 |
| `data` | `Dict[str, Any]` | 默认 `data` 节 | 合并完整嵌套 data 节。 | 必须为字典。 |
| `loss`, `loss_name` | `Optional[Union[str, nn.Module]]` | `None` | `loss.name` 的平铺别名；省略时可使用模型默认 loss。 | 注册字符串、module 或模型支持的默认值。 |
| `loss_params` | `Dict[str, Any]` | `{}` | 映射到 `loss.params`。 | 传给 loss 构造器。 |
| `output_index` | `Optional[int]` | `None` | tuple/list 模型输出的 `loss.output_index`。 | 必须选择存在的输出。 |
| `output_key` | `Optional[str]` | `None` | mapping 模型输出的 `loss.output_key`。 | 必须选择存在的键。 |
| `optimizer`, `optimizer_name` | `str` | `"adam"` | `optimizer.name` 的平铺别名。 | Trainer 解析支持的名称。 |
| `lr` | `float` | `1e-4` | 映射到 `optimizer.lr`。 | 必须大于 `0`；拒绝布尔值。 |
| `optimizer_params` | `Dict[str, Any]` | `{}` | 映射到 `optimizer.params`。 | 传给 optimizer 构造器。 |
| `scheduler`, `scheduler_name` | `Optional[str]` | `None` | `scheduler.name` 的平铺别名；`None` 关闭 scheduler。 | Trainer 解析支持的名称。 |
| `scheduler_params` | `Dict[str, Any]` | `{}` | 映射到 `scheduler.params`。 | 传给 scheduler 构造器。 |
| `epochs` | `int` | `100` | 映射到 `train.epochs`。 | 正的非布尔整数。 |
| `output_dir` | `Optional[Union[str, Path]]` | `None` | 映射到 `train.output_dir`。 | `None` 解析为 `checkpoints/<Model>_<Dataset>`。 |
| `save_every` | `int` | `0` | 映射到 `train.save_every`；大于零时，按照对应 epoch 间隔额外保存 `epoch_<epoch>.pt`。 | 非负、非布尔整数；`0` 关闭编号快照。 |
| `validate_every` | `int` | `1` | 映射到 `train.validate_every`。 | 正的非布尔整数。 |
| `log_every` | `int` | `10` | 映射到 `train.log_every`。 | 正的非布尔整数。 |
| `grad_clip` | `Optional[float]` | `None` | 映射到 `train.grad_clip`。 | 设置时必须大于 `0`。 |
| `amp` | `bool` | `False` | 映射到 `train.amp`；仅 CUDA 上生效。 | — |
| `resume`, `resume_path` | `Optional[Union[str, Path]]` | `None` | `train.resume` 的平铺别名。 | checkpoint 必须可由 Trainer 加载。 |
| `strict_resume` | `bool` | `True` | 映射到 `train.strict_resume`。 | 控制 state dictionary 严格加载。 |
| `seed` | `Optional[int]` | `42` | 映射到 `train.seed`；`None` 不设置随机种子。 | 整数或 `None`；拒绝布尔值。 |
| `device` | `Union[str, torch.device]` | 最佳可用设备 | 映射到 `train.device`；默认依次选择 CUDA、MPS、CPU。 | 请求不可用 CUDA/MPS 抛 `RuntimeError`。 |
| `device_ids` | `Optional[Union[List[int], Tuple[int, ...]]]` | `None` | 映射到 `train.device_ids`；两个或更多 CUDA 序号启用单进程 `DataParallel`。 | 必须是非空、无重复、非负且可用的 CUDA 序号；第一项必须与显式带序号的 `device` 一致。 |
| `progress_bar` | `bool` | `True` | 映射到 `train.progress_bar`。 | 控制 tqdm 训练/验证进度条。 |
| `model`、`loss`、`optimizer`、`scheduler`、`train`（字典形式） | `Dict[str, Any]` | 对应默认节 | 这些名称的值为字典时，直接合并进对应嵌套节，而非作为平铺别名。 | 合并后各节必须为字典。 |

任何未知的平铺 `**kwargs` 键都会抛 `TypeError`，不会静默忽略。

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

若验证从未产生 loss，`best_val_loss` 保持为正无穷。

## Behavior Details

- 配置优先级为默认值、`config`、平铺/嵌套 `**kwargs` 覆盖。
- Trainer 管理设备；在 `train()` 开始前构建模型、数据集、loss、optimizer、scheduler、AMP scaler、输出目录及可选恢复状态。
- `device_ids` 含两个或更多序号时，Trainer 使用单机单进程 `torch.nn.DataParallel`。原始模型仍负责优化和 checkpoint，因此保存的状态字典键不会增加 `module.` 前缀。
- 每个 epoch 后都会覆盖写入 `last.pt` 和 `best.pt`，与 `save_every` 无关。`save_every > 0` 时，每当 epoch 序号能被它整除，额外保存 `epoch_<epoch>.pt`。
- `ZeroDCEPlusPlus` 等模型配置名可解析带标点的 `ZeroDCE++.yaml`。

### Raises

| Exception | Condition |
| --- | --- |
| `TypeError` | config/节类型非法、未知平铺键、模型类/输入非法、seed/设备序号类型非法或训练对象不兼容。 |
| `ValueError` | 缺少模型/数据集根目录、正数/非负设置非法、学习率/梯度裁剪非正、设备文本非法或 CUDA 设备序号非法/不可用。 |
| `FileNotFoundError` | YAML/checkpoint 路径或内置配置无法解析。 |
| `RuntimeError` | 显式请求的 CUDA/MPS 不可用，或在非 CUDA 设备上使用了 `device_ids`。 |
| `FloatingPointError` | 计算的 loss 为 NaN 或无穷。 |

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

- 模型组件 reference：[`models/`](models/)
- 预测：[`openLLV.predict()`](predict.md)
