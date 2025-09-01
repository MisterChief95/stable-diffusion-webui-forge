# reference: https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/download_supported_configs.py

import os
import shutil

from huggingface_hub import snapshot_download

DIFFUSERS_DEFAULT_PIPELINE_PATHS = [
    {"pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers"},
    {"pretrained_model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"},
    {"pretrained_model_name_or_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
]

for config in DIFFUSERS_DEFAULT_PIPELINE_PATHS:
    try:
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        local_dir = os.path.join("backend", "huggingface", pretrained_model_name_or_path)
        os.makedirs(local_dir, exist_ok=True)

        snapshot_download(pretrained_model_name_or_path, local_dir=local_dir, allow_patterns=["*.json", "*.txt"], token=None, force_download=True)

        shutil.rmtree(os.path.join(local_dir, ".cache"))

        _files = []
        for dirpath, _, filenames in os.walk(local_dir):
            for filename in filenames:
                if filename.endswith(".safetensors.index.json"):
                    os.remove(os.path.join(dirpath, filename))
                elif filename.endswith((".json", ".txt")):
                    _files.append(os.path.join(dirpath, filename))

        for file in _files:
            with open(file, "r", newline="\n", encoding="utf-8") as infile:
                lines = infile.readlines()
            with open(file, "w", newline="\r\n", encoding="utf-8") as outfile:
                outfile.writelines(lines)

        print(pretrained_model_name_or_path)
    except Exception as e:
        print(e)
