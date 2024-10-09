import numpy as np
import torch
import torch.nn.functional as F


def prepare_image_as_tensor(image, image_size, device):
    if isinstance(image, np.ndarray):
        # convert numpy array (H, W, C) to tensor (C, H, W) and normalize to [0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image_tensor = image.float() / 255.0 if image.max() > 1 else image
        else:
            raise ValueError("Tensor must have 3 dimensions (C, H, W)")
    else:
        raise ValueError("Unsupported image type")

    # resize if necessary
    if image_tensor.shape[1:] != image_size:
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image_tensor.to(device)
