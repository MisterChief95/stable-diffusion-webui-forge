# reference: https://github.com/comfyanonymous/ComfyUI/blob/v0.3.48/comfy/latent_formats.py

class LatentFormat:
    scale_factor = 1.0
    latent_channels = 4
    latent_rgb_factors = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor


class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            #      R        G        B
            [ 0.3512,  0.2297,  0.3227],
            [ 0.3250,  0.4974,  0.2350],
            [-0.2829,  0.1762,  0.2721],
            [-0.2120, -0.2616, -0.7177],
        ]
        self.taesd_decoder_name = "taesd_decoder"


class SDXL(LatentFormat):
    scale_factor = 0.13025

    def __init__(self):
        self.latent_rgb_factors = [
            #      R        G        B
            [ 0.3651,  0.4232,  0.4341],
            [-0.2533, -0.0042,  0.1068],
            [ 0.1076,  0.1111, -0.0362],
            [-0.3165, -0.2492, -0.2188],
        ]
        self.taesd_decoder_name = "taesdxl_decoder"


class Flux(LatentFormat):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0346,  0.0244,  0.0681],
            [ 0.0034,  0.0210,  0.0687],
            [ 0.0275, -0.0668, -0.0433],
            [-0.0174,  0.0160,  0.0617],
            [ 0.0859,  0.0721,  0.0329],
            [ 0.0004,  0.0383,  0.0115],
            [ 0.0405,  0.0861,  0.0915],
            [-0.0236, -0.0185, -0.0259],
            [-0.0245,  0.0250,  0.1180],
            [ 0.1008,  0.0755, -0.0421],
            [-0.0515,  0.0201,  0.0011],
            [ 0.0428, -0.0012, -0.0036],
            [ 0.0817,  0.0765,  0.0749],
            [-0.1264, -0.0522, -0.1103],
            [-0.0280, -0.0881, -0.0499],
            [-0.1262, -0.0982, -0.0778],
        ]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor
