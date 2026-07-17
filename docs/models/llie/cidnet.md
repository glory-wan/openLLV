# HVI-CIDNet

> Task: low-light image enhancement (`llie`)

HVI-CIDNet is the Color and Intensity Decoupling Network from **HVI: A New Color Space for Low-light Image Enhancement** (CVPR 2025). It converts RGB images into a learnable HVI color space, processes chromatic H/V features and intensity features in two interacting branches, and couples them with Lighten Cross-Attention (LCA) blocks.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/abs/2502.20272 |
| Official source code | https://github.com/Fediory/HVI-CIDNet |
| Default configuration | `openLLV/deepLearning/config/CIDNet.yaml` |

## openLLV implementation

| Item | Location |
| --- | --- |
| Model | `openLLV/deepLearning/models/LLIE/CIDNet.py` |
| Model name | `CIDNet` (alias: `HVI-CIDNet`) |
| Loss | `openLLV/deepLearning/loss/LLIELoss/CIDNet_Loss.py` |
| Loss name | `cidnet` |

The integrated loss follows the official training objective in both RGB and HVI domains:

```text
L = L_RGB + hvi_weight * L_HVI
L_domain = pixel_weight * L1
         + ssim_weight * (1 - SSIM)
         + edge_weight * L_edge
         + perceptual_weight * L_VGG19
```

The default weights are `1.0`, `0.5`, `50.0`, `0.01`, and `hvi_weight=1.0`, matching the upstream defaults. VGG19 perceptual features use `conv1_2`, `conv2_2`, `conv3_4`, and `conv4_4` with MSE distance. The first training run may download the ImageNet VGG19 weights through torchvision. Set `loss.params.use_perceptual: false` for an offline or lightweight smoke test.

CIDNet pads tensors internally to a multiple of eight and crops its output back to the original size, so prediction supports arbitrary image dimensions. `input_gamma`, `saturation_scale`, and `intensity_scale` expose the upstream inference controls; each defaults to `1.0`.

## Training

Edit the dataset path in the YAML file, then run:

```bash
python -m openLLV.cli train openLLV/deepLearning/config/CIDNet.yaml
```

or use Python:

```python
import openLLV as llv

result = llv.train("CIDNet", root_dir="datasets/my_dataset")
print(result["checkpoint_dir"])
```

The packaged config uses `CommonDataset`, whose preferred paired directories are `train/input`, `train/target`, `val/input`, and `val/target`. Other registered datasets can be selected through `data.dataset`.

## Prediction

Use a checkpoint produced by the openLLV trainer:

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "checkpoints/CIDNet_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/CIDNet/output.png",
    device="cuda",
)
```

To tune the optional upstream-style inference controls, pass checkpoint configuration overrides through `Predictor`, or create the model with `LLVModel.create_model("CIDNet", config={...})`.

## Official raw weights

The architecture keeps the official parameter names, so a raw upstream `.pth` state dictionary can be loaded into a newly created model:

```python
import torch
from openLLV.deepLearning.models import LLVModel

model = LLVModel.create_model("CIDNet")
state = torch.load("LOLv1.pth", map_location="cpu")
model.load_state_dict(state, strict=True)
model.to("cuda").eval_mode()
```

openLLV training checkpoints contain additional configuration and optimizer metadata; use those checkpoints directly with `llv.predict`.
