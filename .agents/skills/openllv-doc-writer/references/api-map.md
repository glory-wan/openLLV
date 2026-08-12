# openLLV 公开 API 索引（已对照源码核实）

> 本文件记录 openLLV 公开功能的签名、源码位置与 kwargs 路由。**所有信息已通过阅读源码核实**，撰写文档时以此为准；若后续源码变动，以源码为准并更新本表。

## 顶层 API（`openLLV/api.py`）

| 方法 | 签名（核实版） | 说明 |
| --- | --- | --- |
| `predict` | `predict(method, source, output=None, **kwargs)` | 统一预测入口 |
| `enhance` | `enhance(method, source, output=None, **kwargs)` | `predict` 的别名 |
| `train` | `train(config=None, **kwargs)` | 构建 `Trainer` 并跑完整训练循环，返回 `Trainer.train()` 的结果 |
| `evaluate` | `evaluate(en_img_dir=_MISSING, ref_img_dir=_MISSING, metrics=None, save_path=None, return_evaluator=False, *, en=_MISSING, ref=_MISSING, **kwargs)` | 目录评估；`en`/`ref` 为 `en_img_dir`/`ref_img_dir` 的向后兼容别名，同参二义时抛 `TypeError`；`return_evaluator=True` 返回 `Evaluator` 实例而非结果 |
| `eval` | `eval(*args, **kwargs)` | `evaluate` 的别名 |
| `imread` | `imread(source, output_format="pil", **kwargs)` | 读取图像，等价于 `read_image` |
| `imwrite` | `imwrite(image, output=None, *, save_format=None, output_name=None, **kwargs)` | 保存图像，等价于 `write_image`，返回 `Path` |
| `read_image` / `write_image` | 同 `imread` / `imwrite` | 别名 |
| `list_models` | `list_models() -> List[str]` | 全部注册模型名+别名（含大小写折叠） |
| `list_algorithms` | `list_algorithms() -> List[str]` | 全部注册算法名+别名 |
| `list_metrics` | `list_metrics() -> List[str]` | 全部注册评估指标名 |
| `list_losses` | `list_losses() -> List[str]` | 全部注册损失名+别名 |
| `list_datasets` | `list_datasets() -> List[str]` | 全部注册数据集名+别名 |
| `list_available` | `list_available() -> Dict[str, List[Dict]]` | 按类别（models/algorithms/metrics/losses/datasets）返回去重后的组件类，每项含 `name` 与 `aliases` |

## `predict` 的 kwargs 路由（核心）

`openLLV/api.py` 中 `_PREDICT_CALL_KWARGS` 集合（已核实，9 个键）：

```python
{"progress_bar", "output_name", "output_ext", "save", "model_kwargs",
 "ext", "timeout", "headers", "verify_ssl"}
```

- **属于该集合的键** → 调用参数，传给 `predictor(source, output=..., **call_kwargs)`，最终到达后端 `predict_single` / `predict_batch`。
- **其余键** → 进 `Predictor` 构造器（`openLLV/predictor.py`）。

### 统一 `Predictor`（`openLLV/predictor.py`）

```python
Predictor(target=None, *, model=None, method=None, backend="auto",
          output_dir=None, config=None, device=None, transform=None,
          batch_size=1, num_workers=0, **kwargs)
```

- `target`：模型名 / checkpoint 路径 / 算法名 / `LLVModel` 实例 / `LLVEnhancer` 实例。
- `model` / `method`：显式选择器，与 `target` 互斥（同时传抛 `ValueError`）。
- `backend`：`"auto"` 或深度别名（`deep`/`deeplearning`/`deep_learning`/`dl`/`model`）或传统别名（`tradition`/`traditional`/`traditionalalgorithm`/`traditional_algorithm`/`ta`/`method`/`algorithm`）；大小写与空白不敏感。`auto` 推断规则：`.pt`/`.pth` → 深度；注册名同时存在于两个注册表 → 抛 `ValueError` 要求显式 `backend`。
- `output_dir`：默认输出目录（单图未给 `output` 时用 `output_dir/<源文件名>`；目录输入时作为输出根目录）。
- 其余 `**kwargs` 行为随后端而异（见下）。
- `Predictor.__call__(source, output=None, **kwargs)` / `predict` 同签名；`predict_single` / `predict_batch` 原样转发给后端。

### 深度后端 `Predictor`（`openLLV/deepLearning/predictor.py`）

构造器：`Predictor(model, output_dir=None, config=None, device=None, transform=None, batch_size=1, num_workers=0)`。

- `model`：注册名 / checkpoint（`.pt`/`.pth`）/ `LLVModel` 实例。
- `config`：模型配置覆盖；对 checkpoint 会覆盖保存的配置。
- `device`：`None` → CUDA 可用则 CUDA，否则 CPU。
- `batch_size` / `num_workers`：目前仅为元数据（预留），目录推理仍逐图处理。
- `predict_single(image, save_path=None, *, output_name=None, output_ext=None, save=True, transform=None, model_kwargs=None, **reader_kwargs)`：`model_kwargs` 转发给模型 `forward`（其中张量值自动搬到设备）；`**reader_kwargs` 转给 `ImageReader`（`ext`、`timeout`、`headers`、`verify_ssl`）。返回 `(PIL.Image, Path|None)`。
- `predict_batch(input_dir, output_dir=None, *, progress_bar=True, transform=None, model_kwargs=None, **reader_kwargs)`：递归处理，保留相对子目录与源后缀，返回按路径排序的 `Path` 列表。

### 传统后端 `Predictor`（`openLLV/tradition/predictor.py`）

构造器：`Predictor(method="he", output_dir=None, config=None, **kwargs)`。

- `method`：注册名或 `LLVEnhancer` 实例。实例会强制 `set_params(output_type="numpy")`。
- `config` + `**kwargs`：合并后传给 `LLVEnhancer.create_enhancer`；**不支持的参数被忽略并告警**。
- `predict_single(image, save_path=None, *, output_name=None, output_ext=None, save=True, **kwargs)`：返回 `(numpy.ndarray, Path|None)`。
- `predict_batch(input_dir, output_dir=None, *, progress_bar=True, **kwargs)`：返回 `Path` 列表。

## `train` 的 kwargs 路由

`Trainer(config=None, **kwargs)`（`openLLV/deepLearning/trainer.py`）。`config` 可为：内置配置名（大小写/标点不敏感）、YAML 路径、嵌套字典、`None`。`**kwargs` 经 `_kwargs_to_config` 的 `flat_map` 映射到嵌套配置，**未知键抛 `TypeError`**。已核实平铺键（节选自 `flat_map`）：

| 键 | 映射目标 |
| --- | --- |
| `model` / `model_name` | `model.name` |
| `model_params` | `model.params` |
| `dataset` / `dataset_name` | `data.dataset` |
| `root_dir` | `data.root_dir` |
| `batch_size` / `num_workers` / `pin_memory` / `shuffle` / `drop_last` | `data.*` |
| `train_split` / `val_split` / `return_filename` | `data.*` |
| `resize` | `data.resize`（int=方形，`[h, w]`=显式尺寸） |
| `train_input_dir` / `train_target_dir` / `val_input_dir` / `val_target_dir` | `data.*` |
| `data_params` / `train_params` / `val_params` | `data.params` / `data.train_params` / `data.val_params` |
| `loss` / `loss_name` / `loss_params` / `output_index` / `output_key` | `loss.*` |
| `optimizer` / `optimizer_name` / `lr` / `optimizer_params` | `optimizer.*` |
| `scheduler` / `scheduler_name` / `scheduler_params` | `scheduler.*`（`name: null` 关闭） |
| `epochs` / `output_dir` / `save_every` / `validate_every` / `log_every` | `train.*` |
| `grad_clip` / `amp` / `seed` / `device` / `progress_bar` | `train.*` |
| `resume` / `resume_path` / `strict_resume` | `train.*` |

`train()` 返回字典含 history、best_val_loss、checkpoint_dir。

## `evaluate` 的 kwargs 路由

kwargs 全部转给 `Evaluator`（`openLLV/evaluation/evaluator.py`）：

```python
Evaluator(en_img_dir, ref_img_dir=None, save_path=None, metrics=None,
          device=None, batch_size=1, num_workers=8, **kwargs)
```

- `metrics`：`None` → `["PSNR", "SSIM"]`；字符串自动转大写；列表逐项大写；其它类型抛 `TypeError`。不存在的指标名告警跳过。
- `device`：`None` → CUDA 可用则 CUDA，否则 CPU。
- `**kwargs` 转给 `BaseMetric.create_metric(name, device=..., **kwargs)`（即各指标构造器参数，如 `pyiqa` 相关选项）。
- `save_path`：`None` → 保存到 `./results/eval.json`。
- 结果字典结构：`{"filenames": [...], "metrics": {名称: {文件名: 值}}, "statistics": {名称: {mean, std, min, max, valid_count, total_count, better}}}`；缺参考图的参考指标记 `NaN`。

## 图像 I/O

- `ImageReader`（`openLLV/data/image_io.py`）：`__call__(input_data, output_format=ImageFormat.PIL, **kwargs)`。`output_format` 取值 `"pil"`（RGB）、`"numpy"`（BGR）、`"bytes"`、`"base64"`、`"file"`（临时文件路径）。kwargs 支持 `ext`（无后缀源数据编码用）、`timeout`（默认 10）、`headers`（默认含 User-Agent）、`verify_ssl`（默认 True）。`get_info()` 返回输入元数据。
- `ImageWriter`（同文件）：`__call__(image, output=None, *, save_format=None, output_name=None, **reader_kwargs)`，返回 `Path`。`output` 省略时用 `results/`；`save_format` 覆盖输出后缀；JPG 输出自动转 RGB。
- 顶层 `imread`/`imwrite` 是上述的薄封装。

## 注册名规则

- 模型注册名：类名 + `aliases` 类属性，`LLVModel._model_registry`，查找时 `strip().lower()`（大小写不敏感）。
- 算法注册名：类 `name` 属性 + 别名，`LLVEnhancer._enhancer_registry`，同样大小写不敏感。
- 组件参数入口：`LLVEnhancer.create_enhancer(name, **kwargs)` 会把不支持的参数过滤掉并告警（与深度后端行为不同——深度后端 kwargs 进模型 `config`）。
- `LLVModel` 构造器：`LLVModel(config=None, **kwargs)`，kwargs 覆盖 config 同名键，最终存入 `self.config`（含 `model_name`、`input_channels`、`save_dir` 默认值）；模型具体参数在 `_init_model()` 中从 `self.config` 读取（如 `input_gamma`、`saturation_scale` 等），**写组件文档时必须逐个读取 `_init_model` 中读取的 config 键**。

## 常规查询命令（撰写时可复现）

```bash
# 签名与默认值
python3 -c "import inspect, openLLV; print(inspect.signature(openLLV.predict))"
# 注册名/别名
python3 -c "import openLLV; print(openLLV.list_available())"
# 某算法构造参数（含基类参数）
python3 -c "from openLLV.tradition.algorithms import LLVEnhancer; print(LLVEnhancer._get_constructor_parameter_names(LLVEnhancer.create_enhancer('clahe').__class__))"
```
