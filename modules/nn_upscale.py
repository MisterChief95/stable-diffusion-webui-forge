import os
import torch
import backend.memory_management as mm
from modules.latent_resizer import LatentResizer
from modules.paths_internal import models_path


NN_VERSIONS = [
    "SDXL NeuralNetwork",
    "SD 1.x NeuralNetwork",
]


class NNLatentUpscale:
    """
    Upscales SDXL latent using neural network
    """

    def __init__(self):
        self.local_dir = os.path.dirname(os.path.realpath(__file__))
        self.scale_factor = 0.13025
        # Use backend's VAE dtype selection for consistency
        device = mm.get_torch_device()
        self.dtype = mm.vae_dtype(device)
        self.weight_path = {
            NN_VERSIONS[0]: os.path.join(models_path, "nn_latent", "sdxl_resizer.pt"),
            NN_VERSIONS[1]: os.path.join(models_path, "nn_latent", "sd15_resizer.pt"),
        }

    def upscale(self, latent: torch.Tensor, version: str, scale: float) -> torch.Tensor:
        device = mm.get_torch_device()
        samples = latent["samples"].to(device=device, dtype=self.dtype)

        model = LatentResizer.load_model(
            self.weight_path[version], device, self.dtype
        )

        try:
            model.to(device=device)
            model.eval()  # Ensure model is in eval mode
            
            # The neural network was trained on raw latent values, not VAE-scaled ones
            with torch.no_grad():
                latent_out = model(samples, scale=scale)

            if self.dtype != torch.float32:
                latent_out = latent_out.to(dtype=torch.float32)
        finally:
            model.cpu()
            del model
            mm.soft_empty_cache()

        return latent_out
