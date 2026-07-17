# Custom Traditional Enhancement Algorithms

Traditional low-level vision methods inherit from `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py`. Place an implementation in the package matching its domain, such as `BaseMethods`, `Dehazing`, or `LLIE`.

## 1. Base Class Contract

`LLVEnhancer` provides image loading, BGR NumPy conversion, validation, dtype preservation, clipping, output conversion, automatic registration, and factory construction. Subclasses implement only `_enhance()` plus their own parameters.

## 2. Minimal Algorithm

```python
from typing import Any

import numpy as np

from openLLV.tradition.algorithms import LLVEnhancer


class MyAlgorithm(LLVEnhancer):
    name = "my_algorithm"
    aliases = ["myalgo"]

    def __init__(self, strength: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        if strength < 0:
            raise ValueError("strength must be non-negative.")
        self.strength = float(strength)

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        strength = float(kwargs.get("strength", self.strength))
        result = image.astype(np.float32) * strength
        return result

    def get_params(self):
        params = super().get_params()
        params["strength"] = self.strength
        return params
```

Three-channel arrays received by `_enhance()` use OpenCV-style BGR order. Return a NumPy array; the base class performs clipping and optional dtype restoration.

## 3. Base Options

| Option | Default | Meaning |
| --- | --- | --- |
| `output_type` | `"numpy"` | `numpy`, `pil`, `bytes`, `base64`, or `file` |
| `keep_dtype` | `True` | Cast the result back to the input dtype |
| `clip_output` | `True` | Clip to the valid dtype range |

## 4. Register and Use

Export the class from its domain `__init__.py` and from `openLLV/tradition/algorithms/__init__.py` so importing openLLV triggers registration.

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "myalgo",
    "input.jpg",
    output="results/myalgo.png",
    strength=1.2,
)
```

Direct factory use is also available:

```python
from openLLV.tradition.algorithms import LLVEnhancer

enhancer = LLVEnhancer.create_enhancer(
    "my_algorithm",
    output_type="pil",
    strength=1.2,
)
result = enhancer("input.jpg")
```

Use `llv.list_algorithms()` to confirm the class name, declared name, and aliases were registered.

