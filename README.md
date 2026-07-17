<div align="center">
  <h1>OpenLLV</h1>


  <p>
    An open and extensible Python toolkit for low-level vision.
  </p>

  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch 2.0+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version 0.1.0"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  </p>

  <p>
    <a href="docs/guide/overview.md">English Docs</a> |
    <a href="docs-zh-CN/guide/overview.md">中文文档</a> |
    <a href="docs/usage/cli.md">CLI</a> |
    <a href="docs/usage/cfg.md">Configuration</a> |
    <a href="TEST_PLAN.md">Test Plan</a>
  </p>
</div>

openLLV is an open-source framework for **low-level vision (LLV)**. It provides
one consistent interface for traditional image-processing algorithms,
deep-learning models, training, prediction, image I/O, and image-quality
evaluation.

The project evolves [LibLLIE](https://github.com/glory-wan/LibLLIE) from a
low-light image enhancement library into a task-oriented low-level vision
toolkit. All learned models inherit directly from `LLVModel`; all traditional
methods inherit from `LLVEnhancer`. A class-level `task` value and the package
layout group implementations by domain without introducing a second base-model
layer.

Version **0.1.0** currently provides complete built-in implementations mainly
for low-light image enhancement (LLIE), together with general histogram-based
enhancement and a Dark Channel Prior dehazing method. Packages for additional
tasks are reserved for future implementations and are not presented as
supported algorithms yet.

> **Project status:** openLLV 0.1.0 is preparing for its first public release.
> Install it from source until the package is published to PyPI.

## Highlights

- Unified top-level API: `predict`, `enhance`, `train`, `evaluate`, `imread`,
  `imwrite`, and component-listing helpers.
- One Predictor interface that automatically routes requests to the
  deep-learning or traditional backend.
- A common `LLVModel` contract and automatic registry for learned low-level
  vision models.
- A common `LLVEnhancer` contract and automatic registry for traditional
  algorithms.
- 17 LLIE model implementations with packaged YAML training configurations and
  corresponding registered losses.
- 13 traditional methods covering base enhancement, LLIE, and dehazing.
- Configuration-driven Trainer with validation, checkpoints, resume, schedulers,
  gradient clipping, and automatic mixed precision.
- Full-reference and no-reference image-quality evaluation.
- Extensible registries for models, losses, datasets, algorithms, and metrics.
- English and Chinese documentation with the same directory structure.

## Current Scope

| Domain | Current built-in support | Status |
| --- | --- | --- |
| Low-light image enhancement (`llie`) | 17 learned models and 8 traditional methods | Implemented |
| General enhancement | HE, AHE, CLAHE, RCLAHE | Implemented |
| Dehazing | Traditional DCP/Dark Channel | Implemented |
| All-in-one restoration, deblurring, denoising, deraining, super-resolution, underwater enhancement, and other LLV tasks | Task-oriented package structure | Planned |

The reserved task packages make it possible to add new domains while keeping
the same public API, registry, Predictor, Trainer, data, and evaluation
infrastructure. An empty task package is not evidence that an algorithm in that
domain has already been implemented.

## Documentation

This README provides a compact introduction. Use the complete documentation for
API contracts, configuration fields, extension guides, and component-specific
details.

| Topic | English | 中文 |
| --- | --- | --- |
| API overview | [docs/guide/overview.md](docs/guide/overview.md) | [docs-zh-CN/guide/overview.md](docs-zh-CN/guide/overview.md) |
| Image I/O | [docs/guide/image_io.md](docs/guide/image_io.md) | [docs-zh-CN/guide/image_io.md](docs-zh-CN/guide/image_io.md) |
| Prediction | [docs/guide/predict.md](docs/guide/predict.md) | [docs-zh-CN/guide/predict.md](docs-zh-CN/guide/predict.md) |
| Training | [docs/guide/train.md](docs/guide/train.md) | [docs-zh-CN/guide/train.md](docs-zh-CN/guide/train.md) |
| Evaluation | [docs/guide/evaluate.md](docs/guide/evaluate.md) | [docs-zh-CN/guide/evaluate.md](docs-zh-CN/guide/evaluate.md) |
| CLI | [docs/usage/cli.md](docs/usage/cli.md) | [docs-zh-CN/usage/cli.md](docs-zh-CN/usage/cli.md) |
| Configuration | [docs/usage/cfg.md](docs/usage/cfg.md) | [docs-zh-CN/usage/cfg.md](docs-zh-CN/usage/cfg.md) |
| Custom components | [docs/custom](docs/custom) | [docs-zh-CN/custom](docs-zh-CN/custom) |

## Installation

openLLV requires **Python 3.8 or later** and **PyTorch 2.0 or later**.

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/glory-wan/openLLV.git
cd openLLV
python -m pip install -e .
```

Install the development dependency when running the test suite:

```bash
python -m pip install -e ".[dev]"
```

The distribution installs two equivalent command names:

```bash
openllv --help
llv --help
```

After the first PyPI release, the regular installation command will be:

```bash
python -m pip install openLLV
```

Some evaluation metrics use `pyiqa` and may initialize metric-specific model
weights. Deep-learning inference with meaningful visual quality also requires
compatible trained weights; the repository does not imply that randomly
initialized networks are pretrained.

## Quick Start

### List registered components

```python
import openLLV as llv

components = llv.list_available()
print(components["models"])
print(components["algorithms"])
print(components["metrics"])
```

Flat helpers return every accepted registry key, including aliases:

```python
print(llv.list_models())
print(llv.list_algorithms())
print(llv.list_metrics())
print(llv.list_losses())
print(llv.list_datasets())
```

The same information is available from the CLI:

```bash
openllv list
```

### Image I/O

```python
import openLLV as llv

image = llv.imread("input.jpg", output_format="pil")
array = llv.imread("input.jpg", output_format="numpy")
saved_path = llv.imwrite(image, "results/copy.png")
```

### Traditional prediction

Pass a registered algorithm name and its constructor parameters:

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma.png",
    gamma=0.8,
)
```

Directory input is processed recursively and returns saved output paths:

```python
saved_paths = llv.predict(
    "RCLAHE",
    "images/",
    output="results/rclahe",
    color_space="hsv",
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    iterations=2,
    progress_bar=True,
)
```

CLI equivalent:

```bash
openllv predict Gamma input.jpg -o results/gamma.png --kwargs gamma=0.8
```

### Deep-learning prediction

For reproducible and meaningful inference, pass an openLLV checkpoint created
by `LLVModel` or `Trainer`:

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/zero_dce.png",
    device="cuda",
)
```

A registered model name can also be used directly:

```python
enhanced, saved_path = llv.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/zero_dce_untrained.png",
    device="cpu",
)
```

In the second example the model is newly initialized. It demonstrates API and
model construction only; it does not provide pretrained enhancement quality.
Raw upstream state dictionaries do not include openLLV model metadata and must
be loaded into the matching model class manually.

The Predictor owns device placement. `LLVModel` deliberately does not store or
manage device state.

### Unified Predictor object

```python
from openLLV import Predictor

predictor = Predictor(
    "Gamma",
    backend="traditional",
    output_dir="results/gamma",
    gamma=0.8,
)

enhanced, saved_path = predictor("input.jpg")
```

Use `backend="deep"` for a learned model. With `backend="auto"`, openLLV
selects the backend from the registered name, instance type, or checkpoint
suffix.

### Training

Train a registered model using a packaged configuration:

```python
import openLLV as llv

result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=4,
    device="cuda",
)

print(result["checkpoint_dir"])
```

The preferred paired dataset layout is:

```text
dataset_root/
  train/
    input/
    target/
  val/
    input/
    target/
```

Training can also be configured entirely with keyword arguments:

```python
result = llv.train(
    model="ZeroDCE",
    model_params={"input_channels": 3},
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="zerodce",
    optimizer="adam",
    lr=1e-4,
    epochs=10,
    batch_size=4,
    device="cuda",
)
```

CLI examples:

```bash
# Select a packaged YAML configuration by model/config name
openllv train ZeroDCE --kwargs root_dir=datasets/my_dataset epochs=10 device=cuda

# Use an explicit YAML file
openllv train configs/experiment.yaml --kwargs lr=5e-5 amp=true

# Resume a previous experiment
openllv train ZeroDCE --kwargs root_dir=datasets/my_dataset resume=checkpoints/ZeroDCE_CommonDataset/checkpoints/last.pt
```

By default, Trainer writes to `checkpoints/<Model>_<Dataset>/`:

```text
checkpoints/<Model>_<Dataset>/
  checkpoints/
    best.pt
    last.pt
  logs/
    history.json
  <Model>.yaml
```

See the [training guide](docs/guide/train.md) and
[configuration reference](docs/usage/cfg.md) for nested configuration,
optimizer, scheduler, AMP, validation, and resume behavior.

### Evaluation

Evaluate a directory with full-reference metrics:

```python
import openLLV as llv

results = llv.evaluate(
    en_img_dir="results/enhanced",
    ref_img_dir="datasets/reference",
    metrics=["PSNR", "SSIM", "LPIPS"],
    save_path="results/evaluation.json",
    device="cpu",
)
```

No-reference metrics omit `ref_img_dir`:

```python
results = llv.evaluate(
    en_img_dir="results/enhanced",
    metrics=["NIQE", "MUSIQ", "PI"],
)
```

CLI equivalent:

```bash
openllv evaluate --en results/enhanced --ref datasets/reference --metrics PSNR SSIM --save-path results/evaluation.json
```

## Supported Components

### Deep-learning models

All current learned models use `task = "llie"` and inherit directly from
`LLVModel`.

| Model | Year | Venue | Documentation | Paper | Official source | Upstream license |
| --- | --- | --- | --- | --- | --- | --- |
| LLNet | 2017 | Pattern Recognition | [docs](docs/models/llie/llnet.md) | [paper](https://doi.org/10.1016/j.patcog.2016.06.008) | [code](https://github.com/kglore/llnet_color) | - |
| KinD | 2019 | ACM MM | [docs](docs/models/llie/kind.md) | [paper](https://doi.org/10.1145/3343031.3350926) | [code](https://github.com/zhangyhuaee/KinD) | - |
| Zero-DCE | 2020 | CVPR | [docs](docs/models/llie/zero-dce.md) | [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf) | [code](https://github.com/Li-Chongyi/Zero-DCE) | CC BY-NC 4.0 |
| Zero-DCE++ | 2021 | IEEE TPAMI | [docs](docs/models/llie/zero-dce++.md) | [paper](https://ieeexplore.ieee.org/document/9369102/) | [code](https://github.com/Li-Chongyi/Zero-DCE_extension) | CC BY-NC 4.0 |
| RUAS | 2021 | CVPR | [docs](docs/models/llie/ruas.md) | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf) | [code](https://github.com/KarelZhang/RUAS) | - |
| KinD++ | 2021 | IJCV | [docs](docs/models/llie/kind++.md) | [paper](https://doi.org/10.1007/s11263-020-01407-x) | [code](https://github.com/zhangyhuaee/KinD_plus) | - |
| EnlightenGAN | 2021 | IEEE TIP | [docs](docs/models/llie/enlightengan.md) | [paper](https://doi.org/10.1109/TIP.2021.3051462) | [code](https://github.com/VITA-Group/EnlightenGAN) | - |
| SCI | 2022 | CVPR | [docs](docs/models/llie/sci.md) | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf) | [code](https://github.com/vis-opt-group/SCI) | - |
| URetinex-Net | 2022 | CVPR | [docs](docs/models/llie/uretinex-net.md) | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf) | [code](https://github.com/AndersonYong/URetinex-Net) | MIT |
| LEDNet | 2022 | ECCV | [docs](docs/models/llie/lednet.md) | [paper](https://arxiv.org/pdf/2202.03373) | [code](https://github.com/sczhou/LEDNet) | S-Lab License 1.0 |
| LLFlow | 2022 | AAAI | [docs](docs/models/llie/llflow.md) | [paper](https://doi.org/10.1609/aaai.v36i3.20162) | [code](https://github.com/wyf0912/LLFlow) | CC BY-NC-SA 4.0 |
| RetinexFormer | 2023 | ICCV | [docs](docs/models/llie/retinexformer.md) | [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.pdf) | [code](https://github.com/caiyuanhao1998/Retinexformer) | - |
| PairLIE | 2023 | CVPR | [docs](docs/models/llie/pairlie.md) | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf) | [code](https://github.com/zhenqifu/PairLIE) | - |
| LLFormer | 2023 | AAAI Oral | [docs](docs/models/llie/llformer.md) | [paper](https://arxiv.org/abs/2212.11548) | [code](https://github.com/TaoWangzj/LLFormer) | CC BY-NC-SA 4.0 |
| Zero-IG | 2024 | CVPR | [docs](docs/models/llie/zero-ig.md) | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf) | [code](https://github.com/Doyle59217/ZeroIG) | - |
| DarkIR | 2025 | CVPR | [docs](docs/models/llie/darkir.md) | [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.pdf) | [code](https://github.com/cidautai/DarkIR) | MIT |
| HVI-CIDNet | 2025 | CVPR | [docs](docs/models/llie/cidnet.md) | [paper](https://arxiv.org/abs/2502.20272) | [code](https://github.com/Fediory/HVI-CIDNet) | MIT |

The license column reports the license stated by each upstream repository. A
`-` means no explicit upstream license has been confirmed by this table; public
source availability alone does not grant reuse rights. Several listed licenses
restrict commercial use and are not OSI-approved open-source licenses. See the
[License](#license-and-upstream-notices) section before redistributing code or
weights.

### Traditional algorithms

| Domain | Algorithm | Documentation |
| --- | --- | --- |
| Base method | HE | [docs](docs/algorithms/base_methods/he.md) |
| Base method | AHE | [docs](docs/algorithms/base_methods/ahe.md) |
| Base method | CLAHE | [docs](docs/algorithms/base_methods/clahe.md) |
| Base method | RCLAHE | [docs](docs/algorithms/base_methods/rclahe.md) |
| Dehazing | DCP / DarkChannel | [docs](docs/algorithms/dehazing/dcp.md) |
| LLIE | Gamma | [docs](docs/algorithms/llie/gamma.md) |
| LLIE | BIMEF | [docs](docs/algorithms/llie/bimef.md) |
| LLIE | GCP | [docs](docs/algorithms/llie/gcp.md) |
| LLIE | LIME | [docs](docs/algorithms/llie/lime.md) |
| LLIE | NPE | [docs](docs/algorithms/llie/npe.md) |
| LLIE | SSR | [Retinex docs](docs/algorithms/llie/retinex.md) |
| LLIE | MSR | [Retinex docs](docs/algorithms/llie/retinex.md) |
| LLIE | MSRCR | [Retinex docs](docs/algorithms/llie/retinex.md) |

### Evaluation metrics

| Type | Metrics | Additional backend |
| --- | --- | --- |
| Full-reference | PSNR, SSIM, MSE, MAE, LPIPS, LOE | LPIPS uses `pyiqa` |
| No-reference | NIQE, MUSIQ, PI | `pyiqa` |

See the [evaluation guide](docs/guide/evaluate.md) and
[custom metric guide](docs/custom/metric.md).

## Architecture and Extension System

openLLV registers major components automatically after their implementation
module is imported.

| Component | Base class | Task classification | Guide |
| --- | --- | --- | --- |
| Learned model | `LLVModel` | `task` class variable | [custom model](docs/custom/model.md) |
| Training loss | `BaseLoss` | package/registry | [custom loss](docs/custom/loss.md) |
| Dataset | `BaseDataset` | registry | [custom dataset](docs/custom/dataset.md) |
| Traditional algorithm | `LLVEnhancer` | package/registry | [custom algorithm](docs/custom/algorithm.md) |
| Evaluation metric | `BaseMetric` | registry | [custom metric](docs/custom/metric.md) |

A new learned model inherits `LLVModel` directly:

```python
from openLLV.deepLearning.models import LLVModel


class MyModel(LLVModel):
    task = "llie"
    aliases = ["my_model"]

    def _init_model(self):
        ...

    def forward(self, x, **kwargs):
        ...
```

A new traditional algorithm inherits `LLVEnhancer`:

```python
from openLLV.tradition.algorithms import LLVEnhancer


class MyAlgorithm(LLVEnhancer):
    name = "my_algorithm"
    aliases = ["myalgo"]

    def _enhance(self, image, **kwargs):
        ...
```

Import the implementation from its task package so registration occurs. Name
and alias conflicts fail explicitly rather than silently replacing an existing
component.

## Project Layout

```text
openLLV/
  api.py                 Top-level convenience API
  cli.py                 Command-line interface
  predictor.py           Unified backend router
  data/                  Image I/O, transforms, and datasets
  deepLearning/
    config/              Packaged YAML training configurations
    loss/                BaseLoss and registered LLIE losses
    models/              LLVModel and task-specific learned models
    predictor.py         Deep-learning inference
    trainer.py           Configuration-driven training
  evaluation/            Evaluator and image-quality metrics
  tradition/
    algorithms/          LLVEnhancer and task-specific methods
    predictor.py         Traditional inference
docs/                    English documentation
docs-zh-CN/              Chinese documentation
examples/                Runnable examples
test/                    Test suite
TEST_PLAN.md             Release-level test plan
```

## Examples

The `examples/` directory contains:

| Script | Purpose |
| --- | --- |
| `00_list_components.py` | List registered components |
| `01_image_io.py` | Read and write several image representations |
| `02_traditional_enhance.py` | Single-image and directory traditional prediction |
| `03_deep_enhance.py` | Deep model construction and prediction |
| `04_evaluate.py` | Prediction followed by image-quality evaluation |
| `05_train_tiny.py` | One-epoch training on a generated tiny dataset |
| `06_model_forward.py` | Direct forward passes through selected models |

Run an example from the repository root:

```bash
python examples/00_list_components.py
python examples/02_traditional_enhance.py
```

Examples use an image from `Inputs/` when available and otherwise create a
small synthetic input. Outputs default to `examples/outputs/`. Override these
locations with `OPENLLV_EXAMPLES_INPUTS` and `OPENLLV_EXAMPLES_OUTPUT`.

## Testing

Run the current test suite:

```bash
python -m pytest -q test
```

Collect tests without executing them:

```bash
python -m pytest --collect-only -q test
```

The release-level testing requirements, model matrix, platform coverage,
packaging checks, CI design, and publication gates are defined in
[TEST_PLAN.md](TEST_PLAN.md). Passing only the existing unit tests is not by
itself sufficient to declare a public release ready.

## Contributing

Contributions are welcome. Useful contributions include:

- implementing a new low-level vision task or model;
- adding traditional restoration algorithms;
- improving model-specific tests and tiny configurations;
- adding reproducible examples or pretrained-weight conversion guidance;
- improving English and Chinese documentation;
- extending datasets, losses, metrics, packaging, and CI coverage.

Keep new components consistent with the registration and base-class contracts.
Every behavior change should include focused tests and updated documentation.
Do not present an implementation as an official reproduction unless its
architecture, preprocessing, weights, and numerical behavior have been
validated against the upstream project.

## License and Upstream Notices

The openLLV project is distributed under the [MIT License](LICENSE).

Individual papers, upstream implementations, pretrained weights, and datasets
may use different terms. The project-level MIT license does not override
non-commercial restrictions, missing upstream licenses, dataset licenses, or
model-weight terms. Review the component documentation and original repository
before commercial use or redistribution.

## Contact

Glory Wan<br>
glory947446@gmail.com

## Citation

Citation metadata for openLLV will be added after the first public release.

```bibtex
@misc{openllv,
  title  = {openLLV: An Open and Extensible Toolkit for Low-Level Vision},
  author = {Wan, Glory and Zhou, Qiyang},
  year   = {2026},
  note   = {Version 0.1.0; citation details to be finalized}
}
```
