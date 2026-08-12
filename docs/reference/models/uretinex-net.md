# URetinex-Net

> Task: low-light image enhancement (LLIE)

URetinex-Net is a Retinex decomposition, deep-unfolding, and illumination-adjustment model registered in openLLV as `URetinexNet`.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| Official source code | https://github.com/AndersonYong/URetinex-Net |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/URetinexNet.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/URetinex.py` |
| Class name | `URetinexNet` |
| Registered name | `URetinexNet` (no aliases; lookup is case-insensitive and trims surrounding whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `URetinex_Loss` in `openLLV/deepLearning/loss/LLIELoss/URetinex_Loss.py` (registered as `uretinex`; aliases `uretinex_loss`, `uretinexnet`, `uretinexnet_loss`, `URetinex_Loss`) |

## Implementation Notes

The constructor is `URetinexNet(config=None, **kwargs)`; keyword overrides take precedence over `config`. Each unfolding round updates reflectance and illumination proxies, then the adjustment module produces `High_L`; the prediction is `High_L * R`. A `ratio` passed to `forward` has highest priority. Otherwise, when `adaptive_ratio=True` and `high_img` is supplied, a second decomposition network computes the ratio; otherwise `adjustment_ratio` is used. Training returns a standardized dictionary with reflectance, illumination, adjusted illumination, and the selected ratio in `aux`; inference returns a tensor. Gradients inside `forward` are enabled only when configured `mode` is `"train"`. The implementation architecture assumes three-channel images; `input_channels` is metadata only.

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Dictionary merged over model defaults. | Non-dictionary, non-`None` values raise `TypeError`; unknown keys are retained but ignored unless consumed. |
| `**kwargs` | `Any` | `{}` | Configuration overrides merged after `config`. | Same key semantics as `config`. |
| `model_name` | `str` | `"URetinexNet"` | Shared model metadata. | Not validated or used to construct the architecture. |
| `input_channels` | `int` | `3` | Shared input-channel metadata. | Must be a positive integer or `ValueError` is raised; the implementation still assumes RGB input. |
| `save_dir` | `str` | `"./checkpoints/llie/URetinexNet"` | Default checkpoint/configuration output directory. | No constructor validation. |
| `unfolding_rounds` | `int` | `3` | Number of unfolding iterations. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `gamma` | `float` | `0.01` | Initial reflectance-proxy penalty. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `lambda` | `float` | `0.01` | Initial illumination-proxy penalty. | Must compare greater than `0`; otherwise `ValueError` is raised. |
| `gamma_offset` | `float` | `0.01` | Per-round increment applied to `gamma` after the first round. | No explicit validation. |
| `lambda_offset` | `float` | `0.01` | Per-round increment applied to `lambda` after the first round. | No explicit validation. |
| `use_concat_l` | `bool` | `True` | Whether the restoration module concatenates illumination with reflectance features. | No explicit validation; truthiness controls architecture construction. |
| `mode` | `str` | `"inference"` | Selects training dictionary output or inference tensor output. | Exactly `"train"` or `"inference"`; otherwise `ValueError` is raised. |
| `adjustment_ratio` | `float` | `5.0` | Fallback illumination adjustment ratio. | No explicit validation. |
| `adaptive_ratio` | `bool` | `False` | Builds and enables the high-image decomposition path. | No explicit validation; truthiness controls architecture construction and forward routing. |

`forward(x, high_img=None, ratio=None)` also accepts `high_img: Optional[torch.Tensor]` and `ratio: Optional[float]`. They are runtime inputs, not stored configuration; `ratio` overrides every configured ratio route.

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "URetinexNet",
    "input.jpg",
    output="results/uretinex/output.png",
    config={"unfolding_rounds": 4, "adjustment_ratio": 4.0},
)
```

```python
import openLLV as llv

result = llv.train(
    "URetinexNet",
    model_params={"mode": "train", "unfolding_rounds": 3},
)
```

## Checkpoint / Official Weights

Pass an openLLV `.pt` or `.pth` checkpoint to `llv.predict`; explicit predictor configuration overrides its saved configuration. No official-weight downloader is implemented.
