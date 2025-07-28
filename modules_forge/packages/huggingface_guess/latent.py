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
            #   R        G        B
            [0.3512, 0.2297, 0.3227],
            [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721],
            [-0.2120, -0.2616, -0.7177],
        ]
        self.taesd_decoder_name = "taesd_decoder"


class SDXL(LatentFormat):
    scale_factor = 0.13025

    def __init__(self):
        self.latent_rgb_factors = [
            #   R        G        B
            [0.3920, 0.4054, 0.4549],
            [-0.2634, -0.0196, 0.0653],
            [0.0568, 0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076],
        ]
        self.taesd_decoder_name = "taesdxl_decoder"


class Flux(LatentFormat):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0404, 0.0159, 0.0609],
            [0.0043, 0.0298, 0.0850],
            [0.0328, -0.0749, -0.0503],
            [-0.0245, 0.0085, 0.0549],
            [0.0966, 0.0894, 0.0530],
            [0.0035, 0.0399, 0.0123],
            [0.0583, 0.1184, 0.1262],
            [-0.0191, -0.0206, -0.0306],
            [-0.0324, 0.0055, 0.1001],
            [0.0955, 0.0659, -0.0545],
            [-0.0504, 0.0231, -0.0013],
            [0.0500, -0.0008, -0.0088],
            [0.0982, 0.0941, 0.0976],
            [-0.1233, -0.0280, -0.0897],
            [-0.0005, -0.0530, -0.0020],
            [-0.1273, -0.0932, -0.0680],
        ]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor
