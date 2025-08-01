import os.path

from modules.shared import opts

FLUX = {
    "svdq-fp4_r32-flux.1-dev.safetensors",
    "svdq-int4_r32-flux.1-dev.safetensors",
    "svdq-fp4_r32-flux.1-kontext-dev.safetensors",
    "svdq-int4_r32-flux.1-kontext-dev.safetensors",
}

if opts.svdq_flux_filename:
    FLUX.update((file.strip() for file in str(opts.svdq_flux_filename).split(",")))

T5 = {
    "awq-int4-flux.1-t5xxl.safetensors",
}

if opts.svdq_t5_filename:
    T5.update((file.strip() for file in str(opts.svdq_t5_filename).split(",")))


def svdq_flux(filenames: list[str]) -> str | None:
    return next((file for file in filenames if os.path.basename(file) in FLUX), None)


def svdq_t5(filenames: list[str]) -> str | None:
    return next((file for file in filenames if os.path.basename(file) in T5), None)
