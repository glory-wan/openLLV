# LEDNet

> Task: low-light image enhancement (`llie`)

LEDNet is a low-light enhancement and deblurring model already implemented in openLLV.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/pdf/2202.03373 |
| Official source code | https://github.com/sczhou/LEDNet |

## Model Introduction

LEDNet targets low-light image enhancement and deblurring tasks. Its design includes multi-scale feature extraction, dynamic convolution, or attention-related modules to improve brightness, details, and clarity in dark scenes at the same time.

In openLLV, LEDNet's configuration includes channel settings, skip connection, side loss, dynamic convolution kernel size, the number of curve-attention iterations, and pyramid pooling bin settings.

## Location in openLLV

| Item | Location |
| --- | --- |
| Model implementation | `openLLV/deepLearning/models/LLIE/LEDNet.py` |
| Model class name | `LEDNet` |
| Default configuration | `openLLV/deepLearning/config/LEDNet.yaml` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/LEDNet_Loss.py` |

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LEDNet",
    "input.jpg",
    output="results/LEDNet/output.png",
    device="cuda",
)
```

Training example:

```python
llv.train(
    model="LEDNet",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="lednet",
    epochs=10,
    batch_size=4,
)
```
