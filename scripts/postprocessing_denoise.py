from modules import scripts_postprocessing, ui_components
import gradio as gr
import cv2
import numpy as np
from PIL import Image


class ScriptPostprocessingDenoise(scripts_postprocessing.ScriptPostprocessing):
    name = "Denoising"
    order = 2049

    def ui(self):
        with ui_components.InputAccordion(False, label="Denoising") as enable:
            denoise_strength = gr.Slider(label='Denoising Strength', value=3.0, minimum=0.1, maximum=10.0, step=0.1, elem_id="postprocess_denoise_strength")
            color_strength = gr.Slider(label='Color Component Strength', value=3.0, minimum=0.1, maximum=10.0, step=0.1, elem_id="postprocess_denoise_color_strength")
            template_window_size = gr.Slider(label='Template Window Size', value=7, minimum=3, maximum=21, step=2, elem_id="postprocess_denoise_template_size")
            search_window_size = gr.Slider(label='Search Window Size', value=21, minimum=7, maximum=35, step=2, elem_id="postprocess_denoise_search_size")
            denoise_method = gr.Dropdown(
                label='Denoising Method',
                choices=['Non-local Means', 'Bilateral Filter', 'Gaussian Blur'],
                value='Non-local Means',
                elem_id="postprocess_denoise_method"
            )
            
        return {
            "d_enable": enable,
            "d_denoise_strength": denoise_strength,
            "d_color_strength": color_strength,
            "d_template_window_size": template_window_size,
            "d_search_window_size": search_window_size,
            "d_denoise_method": denoise_method,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, d_enable, d_denoise_strength, d_color_strength, d_template_window_size, d_search_window_size, d_denoise_method):
        if not d_enable:
            return

        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(pp.image), cv2.COLOR_RGB2BGR)
        
        # Apply selected denoising method
        if d_denoise_method == "Non-local Means":
            denoised = cv2.fastNlMeansDenoisingColored(
                cv_image,
                None,
                h=d_denoise_strength,
                hColor=d_color_strength,
                templateWindowSize=int(d_template_window_size),
                searchWindowSize=int(d_search_window_size)
            )
        elif d_denoise_method == "Bilateral Filter":
            denoised = cv2.bilateralFilter(
                cv_image,
                d=int(d_search_window_size),
                sigmaColor=d_denoise_strength * 2,
                sigmaSpace=d_denoise_strength * 2
            )
        elif d_denoise_method == "Gaussian Blur":
            kernel_size = int(d_template_window_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            denoised = cv2.GaussianBlur(
                cv_image,
                (kernel_size, kernel_size),
                sigmaX=d_denoise_strength / 3.0
            )
        
        # Convert back to PIL format
        denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        pp.image = Image.fromarray(denoised_rgb)