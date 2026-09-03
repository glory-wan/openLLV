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
          output_dir=None, config=None, device=None, transform=None, resize=None,
          batch_size=1, num_workers=0, **kwargs)
```

- `target`：模型名 / checkpoint 路径 / 算法名 / `LLVModel` 实例 / `LLVEnhancer` 实例。
- `model` / `method`：显式选择器，与 `target` 互斥（同时传抛 `ValueError`）。
- `backend`：`"auto"` 或深度别名（`deep`/`deeplearning`/`deep_learning`/`dl`/`model`）或传统别名（`tradition`/`traditional`/`traditionalalgorithm`/`traditional_algorithm`/`ta`/`method`/`algorithm`）；大小写与空白不敏感。`auto` 推断规则：`.pt`/`.pth` → 深度；注册名同时存在于两个注册表 → 抛 `ValueError` 要求显式 `backend`。
- `output_dir`：默认输出目录（单图未给 `output` 时用 `output_dir/<源文件名>`；目录输入时作为输出根目录）。
- `resize`：仅深度后端；`None` 不缩放，正整数为正方形，二元 tuple/list 按 `(height, width)`。传统后端收到非 `None` 值时抛 `ValueError`。
- 其余 `**kwargs` 行为随后端而异（见下）。
- `Predictor.__call__(source, output=None, **kwargs)` / `predict` 同签名；`predict_single` / `predict_batch` 原样转发给后端。

### 深度后端 `Predictor`（`openLLV/deepLearning/predictor.py`）

构造器：`Predictor(model, output_dir=None, config=None, device=None, transform=None, resize=None, batch_size=1, num_workers=0)`。

- `model`：注册名 / checkpoint（`.pt`/`.pth`）/ `LLVModel` 实例。
- `config`：模型配置覆盖；对 checkpoint 会覆盖保存的配置。
- `device`：`None` → CUDA 可用则 CUDA，否则 CPU。
- `resize=None`：默认预处理只将 PIL 图像转换为浮点张量，不缩放；正整数产生正方形输入，二元 tuple/list 按 `(height, width)`，并在自定义 `transform` 前执行。
- `batch_size`：目录输入按源尺寸分组（设置 `resize` 时按目标尺寸分组），只有完整的同尺寸组进行一次模型前向；余数及变换后尺寸不兼容的样本逐图前向，不做 padding。
- `num_workers`：实际传给 DataLoader，用于目录图像读取和 CPU 预处理；模型推理仍在主预测器进程。
- `predict_single(image, save_path=None, *, output_name=None, output_ext=None, save=True, transform=None, model_kwargs=None, **reader_kwargs)`：`model_kwargs` 转发给模型 `forward`（其中张量值自动搬到设备）；`**reader_kwargs` 转给 `ImageReader`（`ext`、`timeout`、`headers`、`verify_ssl`）。返回 `(PIL.Image, Path|None)`。
- `predict_batch(input_dir, output_dir=None, *, progress_bar=True, output_name=None, output_ext=None, save=True, transform=None, model_kwargs=None, **reader_kwargs)`：递归处理并保留相对子目录。`output_name` 非 `None` 抛 `ValueError`；未指定 `output_ext` 时逐字符保留源文件名和后缀（含大小写），指定时为全部文件替换后缀并保留参数大小写。`save=True` 返回按源路径排序的 `Path` 列表；`save=False` 不创建输出文件/目录并返回同序 `PIL.Image` 列表。

### 传统后端 `Predictor`（`openLLV/tradition/predictor.py`）

构造器：`Predictor(method="he", output_dir=None, config=None, **kwargs)`。

- `method`：注册名或 `LLVEnhancer` 实例。实例会强制 `set_params(output_type="numpy")`。
- `config` + `**kwargs`：合并后传给 `LLVEnhancer.create_enhancer`；不支持的参数会被忽略，同时控制台逐项说明其不会被使用且不影响计算；近似合法名称会附带拼写建议。
- `LLVEnhancer` 基类构造参数：`output_type="numpy"`、`keep_dtype=True`、`clip_output=True`、`value_range="auto"`。`value_range` 还接受 `"unit"`、`"byte"` 或两个有限递增数值组成的 tuple/list。
- `value_range="auto"`：`uint8` 使用 `[0,255]`，其它整数使用 `[0, dtype.max]`；浮点最大值 `<= 1` 推断为 `[0,1]`，否则最大值 `<= 255` 推断为 `[0,255]`。自动模式拒绝负值和大于 `255` 的浮点值（可改用显式自定义值域）；非有限浮点值始终拒绝；最大值 `<= 1` 的字节值域浮点图须显式传 `"byte"`。
- `predict_single(image, save_path=None, *, output_name=None, output_ext=None, save=True, **kwargs)`：返回 RGB `(numpy.ndarray, Path|None)`；OpenCV BGR 仅用于算法内部。
- `predict_batch(input_dir, output_dir=None, *, progress_bar=True, output_name=None, output_ext=None, save=True, **kwargs)`：目录命名、后缀、保存和返回规则与深度后端一致；`save=False` 返回同序 RGB `numpy.ndarray` 列表。其余 `**kwargs` 只转发给增强算法。

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

- `ImageReader`（`openLLV/data/image_io.py`）：`__call__(input_data, output_format=ImageFormat.PIL, **kwargs)`。`output_format` 取值 `"pil"`（RGB）、`"numpy"`（RGB）、`"bytes"`、`"base64"`、`"file"`（临时文件路径）；所有公共三通道输入输出统一为 RGB。kwargs 支持 `ext`（无后缀源数据编码用）、`timeout`（默认 10）、`headers`（默认含 User-Agent）、`verify_ssl`（默认 True）。`get_info()` 返回输入元数据。
- `ImageWriter`（同文件）：`__call__(image, output=None, *, save_format=None, output_name=None, **reader_kwargs)`，返回 `Path`。`output` 省略时用 `results/`；`save_format` 覆盖输出后缀；JPG 输出自动转 RGB。
- 顶层 `imread`/`imwrite` 是上述的薄封装。

## 注册名规则

- 模型注册名：类名 + `aliases` 类属性，`LLVModel._model_registry`，查找时 `strip().lower()`（大小写不敏感）。
- 算法注册名：类 `name` 属性 + 别名，`LLVEnhancer._enhancer_registry`，同样大小写不敏感。
- 组件参数入口：`LLVEnhancer.create_enhancer(name, **kwargs)` 会过滤不支持的参数，在控制台逐项说明其不会被使用且不影响计算，并为近似合法名称附带拼写建议（与深度后端行为不同——深度后端 kwargs 进模型 `config`）。所有传统算法都继承基类参数 `output_type`、`keep_dtype`、`clip_output`、`value_range`。
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
