# Custom Deep-Learning Models

Every learned low-level vision model inherits directly from `LLVModel`. Task grouping is declared by the class-level `task` value and reflected in the package directory; there is no task-specific intermediate base class.

## 1. Base Class Contract

The base class is defined in `openLLV/deepLearning/models/BaseModel.py` and provides:

- automatic case-insensitive registration by class name and `aliases`;
- default/config/keyword merging;
- model construction from a registry name;
- checkpoint and YAML serialization;
- a standard `{"pred", "aux", "meta"}` output helper.

Concrete models must implement `_init_model()` and `forward()`.

## 2. Minimal Model

Create the model inside its task package, for example `openLLV/deepLearning/models/LLIE/MyModel.py`:

```python
from typing import Any

import torch
from torch import nn

from openLLV.deepLearning.models import LLVModel


class MyModel(LLVModel):
    task = "llie"
    aliases = ["my_model"]

    def _get_default_config(self):
        config = super()._get_default_config()
        config.update({
            "hidden_channels": 32,
            "mode": "inference",
        })
        return config

    def _validate_config(self):
        super()._validate_config()
        if self.config["hidden_channels"] <= 0:
            raise ValueError("hidden_channels must be positive.")

    def _init_model(self):
        channels = self.config["hidden_channels"]
        self.network = nn.Sequential(
            nn.Conv2d(self.config["input_channels"], channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, self.config["input_channels"], 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, **kwargs: Any):
        pred = self.network(x)
        if self.training:
            return self._format_output(pred, aux={"input": x})
        return pred
```

Import the class from the task package's `__init__.py` and then from `openLLV/deepLearning/models/__init__.py`. Registration occurs when the module is imported.

## 3. Configuration

`LLVModel.__init__` merges values in this order:

1. `_get_default_config()`;
2. the `config` dictionary;
3. direct keyword arguments.

```python
from openLLV.deepLearning.models import LLVModel

model = LLVModel.create_model(
    "my_model",
    config={"hidden_channels": 64},
)
```

Do not store runtime device state in the model configuration. The predictor and trainer call `model.to(device)` and own placement.

## 4. Output Convention

Inference may return a prediction tensor directly. Structured training output should use:

```python
return self._format_output(
    pred,
    aux={"feature": feature},
    meta={"scale": 1},
)
```

The trainer extracts `pred` for supervised losses and preserves `aux` for model-specific/reference-free losses.

## 5. Prediction and Training

After the module is imported:

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "my_model",
    "input.jpg",
    device="cuda",
)
```

Train from flat overrides:

```python
result = llv.train(
    model="my_model",
    model_params={"hidden_channels": 64},
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="l1",
    epochs=10,
)
```

Check registration with `llv.list_models()`. Name conflicts with another model class or alias raise an error during import.

