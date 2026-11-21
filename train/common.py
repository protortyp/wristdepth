"""Model input and depth conversion shared by training and evaluation."""

import numpy as np
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_input(rgb: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(rgb).to(device).permute(2, 0, 1)[None].float() / 255
    return (x - MEAN.to(device)) / STD.to(device)


def predict_depth(model, x: torch.Tensor, f_px: float) -> torch.Tensor:
    inv = model(x)[:, 0]
    return f_px / (x.shape[-1] * inv.float().clamp_min(1e-6))
