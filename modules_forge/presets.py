from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from modules.options import OptionInfo

from modules.shared_items import list_samplers, list_schedulers
import gradio as gr
from enum import Enum


class PresetArch(Enum):
    sd = 1
    xl = 2
    flux = 3


SAMPLERS = {
    PresetArch.sd: "Euler a",
    PresetArch.xl: "DPM++ 2M SDE",
    PresetArch.flux: "Euler",
}

SCHEDULERS = {
    PresetArch.sd: "Automatic",
    PresetArch.xl: "Karras",
    PresetArch.flux: "Beta",
}


def register(options_templates: dict, options_section: Callable, OptionInfo: "OptionInfo"):
    for arch in PresetArch:
        sampler, scheduler = SAMPLERS[arch], SCHEDULERS[arch]

        options_templates.update(
            options_section(
                (f"ui_{arch.name}", arch.name.upper(), "presets"),
                {
                    f"{arch.name}_t2i_sampler": OptionInfo(sampler, "txt2img sampler", gr.Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{arch.name}_t2i_scheduler": OptionInfo(scheduler, "txt2img scheduler", gr.Dropdown, lambda: {"choices": list_schedulers()}),
                    f"{arch.name}_i2i_sampler": OptionInfo(sampler, "img2img sampler", gr.Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{arch.name}_i2i_scheduler": OptionInfo(scheduler, "img2img scheduler", gr.Dropdown, lambda: {"choices": list_schedulers()}),
                },
            )
        )
