# openLLV.predict()

`openLLV.predict()` 使用注册模型、模型 checkpoint 或传统算法执行推理。单图输入返回图像/路径二元组；目录输入返回保存路径，或在 `save=False` 时返回增强图。`openLLV.enhance()` 是其别名。

## Function Form

```python
openLLV.predict(method, source, output=None, **kwargs)
```

- `method`（位置参数）：模型名、checkpoint 路径、算法名、`LLVModel` 实例或 `LLVEnhancer` 实例。
- `source`（位置参数）：`ImageReader` 接受的图像输入，或图像目录。
- `output`（关键字）：可选输出文件（单图）或输出目录（目录输入）。
- `**kwargs`：预测器构造选项、模型/算法参数或预测调用选项，见 [kwargs 路由](#kwargs-路由)。

## Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `method` | `str` / `Path` / `LLVModel` / `LLVEnhancer` | 必填 | 注册模型/算法名、checkpoint 路径（`.pt`/`.pth`）或后端实例 |
| `source` | 任意 `ImageReader` 输入，或 `str`/`Path` 目录 | 必填 | 图像来源或目录；目录会被递归处理 |
| `output` | `Optional[Union[str, Path]]` | `None` | 输出文件（单图）或输出目录（目录输入）。省略时使用后端的 `output_dir`（默认 `results/<名称>`） |
| `backend` | `str` | `"auto"` | `"auto"`、深度后端别名或传统后端别名，见 [后端解析](#后端解析) |
| `output_dir` | `Optional[Union[str, Path]]` | `None` | 省略 `output` 时的默认输出目录 |
| `config` | `Optional[Dict[str, Any]]` | `None` | 模型或算法配置，传给后端 |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | 仅传统算法：输入值域解释。自动推断通常的浮点 `[0,1]` 或 `[0,255]`；显式/自定义值域会被校验 |
| `device` | `Optional[Any]` | `None` | 深度后端设备；`None` → CUDA 可用则 CUDA，否则 CPU |
| `transform` | `Optional[Any]` | `None` | 深度后端输入变换（可调用对象或 torchvision v2 变换列表） |
| `batch_size` | `int` | `1` | 深度后端预留元数据；必须为正整数 |
| `num_workers` | `int` | `0` | 深度后端数据加载器预留元数据；必须非负 |
| `progress_bar` | `bool` | `True` | 目录输入时显示 tqdm 进度条（经 `**kwargs`） |
| `output_name` | `Optional[str]` | `None` | 单图文件名覆盖。保存到目录且为 `None` 时，保留推断出的源名称与后缀及其大小写。目录输入只允许 `None`；任何字符串均抛 `ValueError` |
| `output_ext` | `Optional[str]` | `None` | 保存结果的后缀覆盖，可带或不带前导点。目录输入时，`None` 逐字符保留每个源后缀及其大小写；显式值替换全部后缀并保留参数中的大小写 |
| `save` | `bool` | `True` | 是否保存结果。单图为 `False` 时返回 `path=None`；目录为 `False` 时不创建输出文件/目录，并返回增强图列表 |
| `model_kwargs` | `Optional[Mapping[str, Any]]` | `None` | 转发给模型 `forward()` 的关键字参数；张量值自动搬到设备（深度后端，经 `**kwargs`） |
| `ext` | `Optional[str]` | `None` | 编码 bytes/base64 输入时使用的源扩展名（经 `**kwargs`） |
| `timeout` | `float` | `10` | 远程来源的 URL 超时（经 `**kwargs`） |
| `headers` | `Dict[str, str]` | 默认 User-Agent 字典 | 远程来源的 HTTP 头（经 `**kwargs`） |
| `verify_ssl` | `bool` | `True` | 远程来源的 SSL 校验（经 `**kwargs`） |
| 其它 `**kwargs` | — | — | 深度后端：模型配置覆盖（合并进 `config`）；传统后端：算法构造参数，不支持的参数被忽略并告警 |

### Aliases

| 别名 | 指向 |
| --- | --- |
| `openLLV.enhance()` | `openLLV.predict()` |

注册名匹配大小写不敏感（配置名还忽略标点）。`Predictor.list_available_models()` 与 `Predictor.list_available_methods()` 可列出全部可用查找键。

## Returns

- **单图**：`(image, saved_path)`。
  - 深度后端：`image` 为 `PIL.Image.Image`。
  - 传统后端：`image` 为 RGB `numpy.ndarray`。
  - `saved_path` 为 `Path`；`save=False` 时为 `None`。
- **目录输入**按确定的源路径顺序返回，并保留相对子目录。
  - `save=True`：返回保存后的 `Path` 列表。
  - `save=False`：不创建输出文件或目录；深度后端返回 `PIL.Image.Image` 列表，传统后端返回 RGB `numpy.ndarray` 列表。

## Behavior Details

### kwargs 路由

`openLLV.predict(**kwargs)` 在 `openLLV/api.py` 中拆分关键字参数：

- 属于 `_PREDICT_CALL_KWARGS` 的键（`progress_bar`、`output_name`、`output_ext`、`save`、`model_kwargs`、`ext`、`timeout`、`headers`、`verify_ssl`）为预测调用选项。
- 其余键用于构造统一 `Predictor`，并继续转发给所选后端预测器。

### 目录输出约定

- `output_name=None` 且 `output_ext=None` 时，逐字符复用每个源相对路径：文件名、后缀及其字母大小写均不改变。
- 显式 `output_ext` 只替换全部源后缀，并保留参数后缀的大小写；文件主名与相对子目录不变。
- `output_name` 仅支持单图。目录输入传任何非 `None` 值都会抛 `ValueError`，不会继续转发给模型、读取器或算法。
- `save=False` 按源路径顺序返回增强图且不执行任何文件系统写入；不会创建 `output`/`output_dir`。若同时传 `output_ext`，仅校验其合法性，不产生文件效果。

### 后端解析

`backend="auto"`（默认）时：

- `.pt`/`.pth` 路径选择深度后端。
- 注册名按注册表查找到各自后端。
- 若名称同时存在于两个注册表，抛 `ValueError`，需显式传 `backend="deep"` 或 `backend="traditional"`。
- `LLVModel` 实例选深度；`LLVEnhancer` 实例选传统。

后端别名：深度 = `deep`/`deeplearning`/`deep_learning`/`dl`/`model`；传统 = `tradition`/`traditional`/`traditionalalgorithm`/`traditional_algorithm`/`ta`/`method`/`algorithm`。匹配大小写与空白不敏感。

### 深度后端细节

- `device` 归预测器所有，`LLVModel` 不存储或管理设备。
- `config` 与其余 `**kwargs` 合并进模型配置。
- openLLV 训练器产出的 checkpoint 包含模型类、配置与状态字典；上游原始 `.pth` 状态字典不含这些元数据，需手动构造模型类加载。

### 传统算法细节

- `config` 与其余 `**kwargs` 传给 `LLVEnhancer.create_enhancer()`；所选算法不支持的参数被忽略并发出 `UserWarning`。
- 逐图参数覆盖也可传给 `Predictor.predict_single()`。
- `value_range="auto"` 会保持通常的浮点 `[0,1]` 与 `[0,255]` 约定。最大值 `<= 1` 的字节值域浮点图存在歧义，应使用 `"byte"`。负值或大于 `255` 的浮点值需要显式有效的自定义值域；非有限输入值始终会被拒绝。

### 统一 `Predictor` 对象

```python
Predictor(
    target=None,
    *,
    model=None,
    method=None,
    backend="auto",
    output_dir=None,
    config=None,
    device=None,
    transform=None,
    batch_size=1,
    num_workers=0,
    **kwargs,
)
```

`target` 接受与顶层 `method` 相同的值；也可以用仅关键字 `model` 选择深度模型，或用 `method` 选择传统 enhancer。`target`、`model`、`method` 是互斥选择器，必须有一个能解析到后端。

统一对象公开以下方法，并委托给所选后端：

| Method | Exact signature | Contract |
| --- | --- | --- |
| `__call__` | `predictor(source, output=None, **kwargs)` | 已存在目录路由到 `predict_batch`；其它来源调用 `predict_single`。 |
| `predict` | `predictor.predict(source, output=None, **kwargs)` | `__call__` 的别名。 |
| `predict_single` | `predictor.predict_single(*args, **kwargs)` | 委托给后端。深度后端精确签名：`(image, save_path=None, *, output_name=None, output_ext=None, save=True, transform=None, model_kwargs=None, **reader_kwargs)`；传统后端精确签名省略 `transform`/`model_kwargs`，其余 kwargs 转给 enhancer。 |
| `predict_batch` | `predictor.predict_batch(*args, **kwargs)` | 委托给后端。深度后端精确签名：`(input_dir, output_dir=None, *, progress_bar=True, output_name=None, output_ext=None, save=True, transform=None, model_kwargs=None, **reader_kwargs)`；传统后端精确签名：`(input_dir, output_dir=None, *, progress_bar=True, output_name=None, output_ext=None, save=True, **kwargs)`。 |
| `get_params` | `predictor.get_params() -> Dict[str, Any]` | 返回 `{"backend": "deep" 或 "traditional", "predictor": <后端参数字典>}`。深度字典含模型、任务、设备、输出目录、batch 元数据、config；传统字典含方法、输出目录、enhancer 参数。 |

类方法 `Predictor.list_available_models()`、`list_available_methods()`、`list_available()` 分别返回模型键、算法键或两个类别。

### Raises

| Exception | Condition |
| --- | --- |
| `TypeError` | `config` 类型非法；`LLVEnhancer` 实例走深度后端（或 `LLVModel` 走传统后端）；后端实例类型非法 |
| `ValueError` | 选择器冲突（`target` 与 `model`/`method` 同传，或 `model` 与 `method` 同传）；注册名二义；后端无法解析；`batch_size` 非正；`num_workers` 为负；`output_ext` 为空；目录输入传非 `None` 的 `output_name` |

## Examples

```python
import openLLV as llv

# 传统算法 + 非默认参数
enhanced, saved_path = llv.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma/output.png",
    gamma=0.6,
)
print(type(enhanced))  # <class 'numpy.ndarray'>
```

```python
# 深度学习模型 + GPU + 不保存
enhanced, saved_path = llv.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/zero_dce/output.png",
    device="cuda",
    save=False,
)
print(saved_path)  # None
```

```python
# forward 调用参数放在 model_kwargs
enhanced, _ = llv.predict(
    "LLFormer",
    "input.jpg",
    save=False,
    model_kwargs={"tile_size": 512, "tile_overlap": 64},
)
```

```python
# 目录输入：按指定大小写统一替换后缀
saved_paths = llv.predict(
    "ZeroDCE",
    "images/",
    output="results/zero_dce",
    output_ext=".PNG",
    progress_bar=True,
)
```

```python
# 目录输入且不写入文件系统
images = llv.predict(
    "Gamma",
    "images/",
    save=False,
    progress_bar=False,
)
```

## Related

- 统一预测器对象：`Predictor`（`openLLV/predictor.py`）
- 后端：`openLLV/deepLearning/predictor.py`、`openLLV/tradition/predictor.py`
- 组件文档：`docs/reference/models/`、`docs/reference/algorithms/`
