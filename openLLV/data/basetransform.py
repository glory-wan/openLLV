import torch
import torchvision

if hasattr(torchvision, "disable_beta_transforms_warning"):
    torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2


ToImage = getattr(v2, 'ToImage', getattr(v2, 'ToImageTensor', v2.PILToTensor))
try:
    ToFloat = v2.ToDtype(torch.float32, scale=True)
except TypeError:
    ToFloat = v2.ConvertDtype(torch.float32) if hasattr(v2, 'ConvertDtype') else v2.ToDtype(torch.float32)

predict_Trans = v2.Compose([
    ToImage(),
    ToFloat,
])

