# openLLV.predict()

`openLLV.predict()` runs inference with a registered model, a model checkpoint, or a traditional algorithm. Single-image input returns an image/path pair; directory input returns saved paths or, with `save=False`, enhanced images. `openLLV.enhance()` is an alias.

## Function Form

```python
openLLV.predict(method, source, output=None, **kwargs)
```

- `method` (positional): model name, checkpoint path, algorithm name, `LLVModel` instance, or `LLVEnhancer` instance.
- `source` (positional): image input accepted by `ImageReader`, or an image directory.
- `output` (keyword): optional output file path (single image) or output directory (directory input).
- `**kwargs`: predictor construction options, model/algorithm parameters, or prediction-call options. See [kwargs routing](#kwargs-routing).

## Parameters

| Parameter        | Type                                               | Default                 | Meaning                                                                                                                                                                      |
| ---------------- | -------------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `method`         | `str` / `Path` / `LLVModel` / `LLVEnhancer`        | required                | Registered model/algorithm name, checkpoint path (`.pt`/`.pth`), or backend instance                                                                                         |
| `source`         | any `ImageReader` input, or `str`/`Path` directory | required                | Image source or directory; directories are processed recursively                                                                                                             |
| `output`         | `Optional[Union[str, Path]]`                       | `None`                  | Output file (single image) or output directory (directory input). When omitted, the backend's `output_dir` (default `results/<name>`) is used                                |
| `backend`        | `str`                                              | `"auto"`                | `"auto"`, a deep-learning backend alias, or a traditional backend alias. See [Backend resolution](#backend-resolution)                                                       |
| `output_dir`     | `Optional[Union[str, Path]]`                       | `None`                  | Default output directory used when `output` is omitted                                                                                                                       |
| `config`         | `Optional[Dict[str, Any]]`                         | `None`                  | Model or algorithm configuration; passed to the backend                                                                                                                      |
| `value_range`    | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | Traditional algorithms only: input value-range interpretation. Auto infers ordinary float `[0,1]` or `[0,255]`; explicit/custom ranges are validated                                      |
| `device`         | `Optional[Any]`                                    | `None`                  | Device for the deep-learning backend; `None` → CUDA if available, else CPU                                                                                                   |
| `transform`      | `Optional[Any]`                                    | `None`                  | Input transform for the deep-learning backend (callable or torchvision v2 transform list)                                                                                    |
| `batch_size`     | `int`                                              | `1`                     | Deep-learning metadata reserved for future batched pipelines; must be a positive integer                                                                                     |
| `num_workers`    | `int`                                              | `0`                     | Deep-learning metadata reserved for data-loader pipelines; must be non-negative                                                                                              |
| `progress_bar`   | `bool`                                             | `True`                  | Show a tqdm progress bar for directory input (via `**kwargs`)                                                                                                                |
| `output_name`    | `Optional[str]`                                    | `None`                  | Single-image filename override. `None` preserves the inferred source name and suffix, including case, when saving to a directory. Directory input requires `None`; any string raises `ValueError`                                         |
| `output_ext`     | `Optional[str]`                                    | `None`                  | Saved-output suffix override, with or without a leading dot. For directory input, `None` preserves every source suffix exactly, including case; an explicit suffix replaces every suffix and preserves the supplied case                     |
| `save`           | `bool`                                             | `True`                  | Save results. For one image, `False` returns `path=None`; for a directory, `False` creates no output files/directories and returns enhanced images                                                                                           |
| `model_kwargs`   | `Optional[Mapping[str, Any]]`                      | `None`                  | Keyword arguments forwarded to the model `forward()`; tensor values are moved to the device automatically (deep backend, via `**kwargs`)                                     |
| `ext`            | `Optional[str]`                                    | `None`                  | Source extension used when encoding byte/base64 inputs (via `**kwargs`)                                                                                                      |
| `timeout`        | `float`                                            | `10`                    | URL timeout for remote sources (via `**kwargs`)                                                                                                                              |
| `headers`        | `Dict[str, str]`                                   | default User-Agent dict | HTTP headers for remote sources (via `**kwargs`)                                                                                                                             |
| `verify_ssl`     | `bool`                                             | `True`                  | SSL verification for remote sources (via `**kwargs`)                                                                                                                         |
| other `**kwargs` | —                                                  | —                       | Deep backend: model configuration overrides (merged into `config`). Traditional backend: algorithm constructor parameters; unsupported parameters are ignored with a warning |

### Aliases

| Alias               | Points to           |
| ------------------- | ------------------- |
| `openLLV.enhance()` | `openLLV.predict()` |

Registered names are matched case-insensitively (and punctuation-insensitively for config names). `Predictor.list_available_models()` and `Predictor.list_available_methods()` list every accepted lookup key.

## Returns

- **Single image**: `(image, saved_path)`.
  - Deep backend: `image` is a `PIL.Image.Image`.
  - Traditional backend: `image` is a RGB `numpy.ndarray`.
  - `saved_path` is a `Path`, or `None` when `save=False`.
- **Directory input** follows deterministic source-path order and preserves relative subdirectories.
  - `save=True`: returns saved `Path` objects.
  - `save=False`: creates no output files or directories and returns `PIL.Image.Image` objects for the deep backend or RGB `numpy.ndarray` objects for the traditional backend.

## Behavior Details

### kwargs routing

`openLLV.predict(**kwargs)` splits keyword arguments in `openLLV/api.py`:

- Keys in `_PREDICT_CALL_KWARGS` (`progress_bar`, `output_name`, `output_ext`, `save`, `model_kwargs`, `ext`, `timeout`, `headers`, `verify_ssl`) are prediction-call options.
- All other keys construct the unified `Predictor`, which forwards them to the selected backend predictor.

### Directory output contract

- With `output_name=None` and `output_ext=None`, every relative source path is reused exactly: the filename, suffix, and their letter case are unchanged.
- An explicit `output_ext` replaces every source suffix while preserving the supplied suffix case; it does not rename stems or relative directories.
- `output_name` is single-image-only. Passing any non-`None` value for directory input raises `ValueError` instead of forwarding it to a model, reader, or algorithm.
- `save=False` returns enhanced images in source-path order and performs no filesystem writes. `output`/`output_dir` is not created; `output_ext`, if supplied, is validated but has no output-file effect.

### Backend resolution

With `backend="auto"` (default):

- `.pt`/`.pth` paths select the deep-learning backend.
- Registered names select their own backend by registry lookup.
- If a name exists in both registries, `ValueError` is raised; pass `backend="deep"` or `backend="traditional"` explicitly.
- `LLVModel` instances select deep; `LLVEnhancer` instances select traditional.

Backend aliases: deep = `deep`/`deeplearning`/`deep_learning`/`dl`/`model`; traditional = `tradition`/`traditional`/`traditionalalgorithm`/`traditional_algorithm`/`ta`/`method`/`algorithm`. Matching is case- and whitespace-insensitive.

### Deep-learning specifics

- `device` is owned by the predictor, never stored by `LLVModel`.
- `config` and remaining `**kwargs` are merged into the model configuration.
- Checkpoints created by the openLLV trainer carry model class, configuration, and state dictionary; raw upstream `.pth` state dictionaries do not and must be loaded manually.

### Traditional-algorithm specifics

- `config` and remaining `**kwargs` are passed to `LLVEnhancer.create_enhancer()`; parameters unsupported by the selected algorithm are ignored with a `UserWarning`.
- Per-image overrides can also be passed to `Predictor.predict_single()`.
- `value_range="auto"` preserves ordinary float `[0,1]` and `[0,255]` conventions. Use `"byte"` for an ambiguous byte-range float image whose maximum is `<= 1`. Negative or float values above `255` require an explicit valid custom range; non-finite inputs are always rejected.

### Unified `Predictor` object

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

`target` accepts the same values as top-level `method`. Alternatively, use the keyword-only `model` selector for a deep model or `method` for a traditional enhancer. `target`, `model`, and `method` are mutually exclusive selectors; one must resolve to a backend.

The unified object exposes these public methods and delegates to the selected backend:

| Method | Exact signature | Contract |
| --- | --- | --- |
| `__call__` | `predictor(source, output=None, **kwargs)` | Routes an existing directory to `predict_batch`; otherwise calls `predict_single`. |
| `predict` | `predictor.predict(source, output=None, **kwargs)` | Alias of `__call__`. |
| `predict_single` | `predictor.predict_single(*args, **kwargs)` | Delegates to the backend. Deep exact backend signature: `(image, save_path=None, *, output_name=None, output_ext=None, save=True, transform=None, model_kwargs=None, **reader_kwargs)`. Traditional exact backend signature omits `transform`/`model_kwargs` and forwards remaining kwargs to the enhancer. |
| `predict_batch` | `predictor.predict_batch(*args, **kwargs)` | Delegates to the backend. Deep exact backend signature: `(input_dir, output_dir=None, *, progress_bar=True, output_name=None, output_ext=None, save=True, transform=None, model_kwargs=None, **reader_kwargs)`. Traditional exact backend signature: `(input_dir, output_dir=None, *, progress_bar=True, output_name=None, output_ext=None, save=True, **kwargs)`. |
| `get_params` | `predictor.get_params() -> Dict[str, Any]` | Returns `{"backend": "deep" or "traditional", "predictor": <backend parameter dictionary>}`. The deep dictionary contains model, task, device, output directory, batch metadata, and config; the traditional dictionary contains method, output directory, and enhancer parameters. |

Class methods `Predictor.list_available_models()`, `list_available_methods()`, and `list_available()` return model keys, algorithm keys, or both categories respectively.

### Raises

| Exception    | Condition                                                                                                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TypeError`  | Invalid `config` type; `LLVEnhancer` passed with deep backend (or `LLVModel` with traditional); invalid backend-instance type                                                                                |
| `ValueError` | Conflicting selectors (`target` with `model`/`method`, or both `model` and `method`); ambiguous registered name; unresolvable backend; `batch_size` not positive; `num_workers` negative; empty `output_ext`; non-`None` `output_name` with directory input |

## Examples

```python
import openLLV as llv

# Traditional algorithm with a non-default parameter
enhanced, saved_path = llv.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma/output.png",
    gamma=0.6,
)
print(type(enhanced))  # <class 'numpy.ndarray'>
```

```python
# Deep-learning model on GPU without saving
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
# Forward-call arguments belong in model_kwargs
enhanced, _ = llv.predict(
    "LLFormer",
    "input.jpg",
    save=False,
    model_kwargs={"tile_size": 512, "tile_overlap": 64},
)
```

```python
# Directory input: replace every suffix with the requested case
saved_paths = llv.predict(
    "ZeroDCE",
    "images/",
    output="results/zero_dce",
    output_ext=".PNG",
    progress_bar=True,
)
```

```python
# Directory input without filesystem output
images = llv.predict(
    "Gamma",
    "images/",
    save=False,
    progress_bar=False,
)
```

## Related

- Unified predictor object: `Predictor` in `openLLV/predictor.py`
- Backends: `openLLV/deepLearning/predictor.py`, `openLLV/tradition/predictor.py`
- Component docs: `models/` and `algorithms/` under `docs/reference/`
