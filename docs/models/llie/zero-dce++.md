# Zero-DCE++

> Task: low-light image enhancement (`llie`)

Zero-DCE++ is a lightweight zero-reference low-light enhancement model already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://ieeexplore.ieee.org/document/9369102/ |
| Official source code | https://github.com/Li-Chongyi/Zero-DCE_extension |
| Official project page | https://li-chongyi.github.io/Proj_Zero-DCE++.html |

## Model Introduction

Zero-DCE++ is a lightweight extension of Zero-DCE. Its goal is to further reduce model complexity while preserving the advantages of zero-reference training. The model estimates enhancement curves through a more compact network structure and is suitable for low-light enhancement applications with strict speed and parameter-count requirements.

In openLLV, Zero-DCE++ keeps the interface style of curve-estimation models. In training mode, it returns enhanced results and curve-related intermediate variables; in inference mode, it returns the enhanced image.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/ZeroDCEPlusPlus.py` |
| Model class name | `ZeroDCEPlusPlus` |
| Default configuration | `openLLV/deepLearning/config/ZeroDCE++.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/ZeroDCE_Loss.py` |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroDCEPlusPlus",
    "input.jpg",
    output="results/ZeroDCEPlusPlus/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    model="ZeroDCEPlusPlus",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="zerodce_extension",
    epochs=10,
    batch_size=4,
)
```
