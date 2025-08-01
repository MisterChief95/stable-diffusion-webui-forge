import os.path

from modules.shared import opts  # TODO

FLUX = {
    "svdq-fp4_r32-flux.1-dev.safetensors",
    "svdq-int4_r32-flux.1-dev.safetensors",
    "svdq-fp4_r32-flux.1-kontext-dev.safetensors",
    "svdq-int4_r32-flux.1-kontext-dev.safetensors",
}

T5 = {
    "awq-int4-flux.1-t5xxl.safetensors",
}


def svdq_flux(filenames: list[str]) -> str | None:
    return next((file for file in filenames if os.path.basename(file) in FLUX), None)


def svdq_t5(filenames: list[str]) -> str | None:
    return next((file for file in filenames if os.path.basename(file) in T5), None)
