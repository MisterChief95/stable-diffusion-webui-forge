import json
import os

import gguf
import safetensors.torch
import torch
from einops import rearrange, repeat

import backend.misc.checkpoint_pickle
from backend.operations_gguf import ParameterGGUF


def read_arbitrary_config(directory):
    config_path = os.path.join(directory, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json file found in the directory: {directory}")

    with open(config_path, "rt", encoding="utf-8") as file:
        config_data = json.load(file)

    return config_data


def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    elif ckpt.lower().endswith(".gguf"):
        reader = gguf.GGUFReader(ckpt)
        sd = {}
        for tensor in reader.tensors:
            sd[str(tensor.name)] = ParameterGGUF(tensor)
    else:
        if safe_load:
            if not "weights_only" in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=backend.misc.checkpoint_pickle)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))


def set_attr_raw(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)


def copy_to_param(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def get_attr_with_parent(obj, attr):
    attrs = attr.split(".")
    parent = obj
    name = None
    for name in attrs:
        parent = obj
        obj = getattr(obj, name)
    return parent, name, obj


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def tensor2parameter(x):
    if isinstance(x, torch.nn.Parameter):
        return x
    else:
        return torch.nn.Parameter(x, requires_grad=False)


def fp16_fix(x):
    # An interesting trick to avoid fp16 overflow
    # Source: https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/1114
    # Related: https://github.com/comfyanonymous/ComfyUI/blob/f1d6cef71c70719cc3ed45a2455a4e5ac910cd5e/comfy/ldm/flux/layers.py#L180

    if x.dtype in [torch.float16]:
        return x.clip(-32768.0, 32768.0)
    return x


def dtype_to_element_size(dtype):
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def nested_compute_size(obj, element_size):
    module_mem = 0

    if isinstance(obj, dict):
        for key in obj:
            module_mem += nested_compute_size(obj[key], element_size)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i in range(len(obj)):
            module_mem += nested_compute_size(obj[i], element_size)
    elif isinstance(obj, torch.Tensor):
        module_mem += obj.nelement() * element_size

    return module_mem


def nested_move_to_device(obj, **kwargs):
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = nested_move_to_device(obj[key], **kwargs)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = nested_move_to_device(obj[i], **kwargs)
    elif isinstance(obj, tuple):
        obj = tuple(nested_move_to_device(i, **kwargs) for i in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(**kwargs)
    return obj


def get_state_dict_after_quant(model, prefix=""):
    for m in model.modules():
        if hasattr(m, "weight") and hasattr(m.weight, "bnb_quantized"):
            if not m.weight.bnb_quantized:
                original_device = m.weight.device
                m.cuda()
                m.to(original_device)

    sd = model.state_dict()
    sd = {(prefix + k): v.clone() for k, v in sd.items()}
    return sd


def beautiful_print_gguf_state_dict_statics(state_dict):
    type_counts = {}
    for k, v in state_dict.items():
        gguf_cls = getattr(v, "gguf_cls", None)
        if gguf_cls is not None:
            type_name = gguf_cls.__name__
            if type_name in type_counts:
                type_counts[type_name] += 1
            else:
                type_counts[type_name] = 1
    print(f"GGUF state dict: {type_counts}")
    return


def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.45/comfy/ldm/common_dit.py#L5"""
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"

    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad

    return torch.nn.functional.pad(img, pad, mode=padding_mode)


def process_img(x, index=0, h_offset=0, w_offset=0):
    """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.45/comfy/ldm/flux/model.py#L198"""
    bs, c, h, w = x.shape
    patch_size = 2  # TODO
    x = pad_to_patch_size(x, (patch_size, patch_size))

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    h_len = (h + (patch_size // 2)) // patch_size
    w_len = (w + (patch_size // 2)) // patch_size

    h_offset = (h_offset + (patch_size // 2)) // patch_size
    w_offset = (w_offset + (patch_size // 2)) // patch_size

    img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
    img_ids[:, :, 0] = img_ids[:, :, 1] + index
    img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
    img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
    return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)
