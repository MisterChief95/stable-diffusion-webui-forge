import logging
import os
import sys

from backend import memory_management

from modules import modelloader, upscaler_utils
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData

from modules_forge.utils import prepare_free_memory


logger = logging.getLogger(__name__)


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

    def do_upscale(self, img, selected_model):
        # Intelligent memory preparation based on model size
        try:
            estimated_mb = modelloader.estimate_model_memory_mb(selected_model)
            logger.info(f"Estimated memory requirement for {selected_model}: {estimated_mb}MB")
            
            # Use targeted memory freeing instead of generic prepare_free_memory()
            current_device = memory_management.get_torch_device()
            if not memory_management.is_device_cpu(current_device):
                free_memory = memory_management.get_free_memory(current_device)
                if free_memory < estimated_mb * 1024 * 1024:
                    logger.info(f"Freeing memory for model loading (need {estimated_mb}MB, have {free_memory / (1024 * 1024):.2f}MB)")
                    memory_management.free_memory(estimated_mb * 1024 * 1024, current_device)
            else:
                prepare_free_memory()  # Fallback for CPU
        except Exception as e:
            logger.warning(f"Memory estimation failed: {e}, using generic memory preparation")
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
        model.to('cpu')
        del model
        model = None

        memory_management.soft_empty_cache(force=True)

        return upscaled_img

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} not found")
        
        # Use memory-aware loading for better resource management
        return modelloader.load_model_with_memory_management(
            path,
            device=memory_management.get_torch_device(),
        )
