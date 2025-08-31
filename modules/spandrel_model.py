import os
import sys

from backend import memory_management

from modules import modelloader, upscaler_utils
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData

from modules_forge.utils import prepare_free_memory


class UpscalerSpandrel(Upscaler):
    """
    General class for Spandrel-supported upscaling models
    """

    def __init__(self, dir_name):
        self.name = "Spandrel"
        self.model_name = "Spandrel"
        self.scalers = []
        self.user_path = dir_name

        super().__init__()

        for file in self.find_models(ext_filter=[".pt", ".pth", ".safetensors"]):
            name = modelloader.friendly_name(file)
            scale = 4
            scaler_data = UpscalerData(name, file, upscaler=self, scale=scale)
            self.scalers.append(scaler_data)

    def cleanup_model(model):
        """Thoroughly clean up a PyTorch model from memory"""
        if model is None:
            return
        
        # Clear gradients and set to eval mode
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
        
        # Move to CPU before deletion
        model.to('cpu')
        

    def do_upscale(self, img, selected_model):
        prepare_free_memory()

        try:
            model = self.load_model(selected_model)
        except Exception as e:
            print(
                f"Unable to load {self.model_name} model {selected_model}: {e}",
                file=sys.stderr,
            )
            return img

        model.to(memory_management.get_torch_device())

        tile_size = getattr(opts, "spandrel_model_tile_size", 192)
        tile_overlap = getattr(opts, "spandrel_model_tile_overlap", 8)

        upscaled_img = upscaler_utils.upscale_with_model(
            model,
            img,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

        # Clean up memory
        self.cleanup_model(model)
        model = None  # Clear the variable reference too

        memory_management.soft_empty_cache(force=True)

        return upscaled_img

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} not found")
        return modelloader.load_spandrel_model(
            path,
            device=memory_management.get_torch_device(),
        )
        # Clear the reference
        del model
        
        # Force garbage collection and cache clearing
        import gc
        gc.collect()
        memory_management.soft_empty_cache(force=True)
