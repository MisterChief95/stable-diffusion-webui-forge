# https://github.com/mcmonkeyprojects/sd-dynamic-thresholding


from lib_dynamic_thresholding.dynthres_core import DynThresh

ENABLE: str = "enable"

FLOAT: str = "FLOAT"
MODEL: str = "MODEL"

MAX: str = "max"
MIN: str = "min"
DEFAULT: str = "default"
STEP: str = "step"

class DynamicThresholdingNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (MODEL,),
                "mimic_scale": (FLOAT, {DEFAULT: 7.0, MIN: 0.0, MAX: 100.0, STEP: 0.5}),
                "threshold_percentile": (FLOAT, {DEFAULT: 1.0, MIN: 0.0, MAX: 1.0, STEP: 0.01}),
                "mimic_mode": (DynThresh.Modes, ),
                "mimic_scale_min": (FLOAT, {DEFAULT: 0.0, MIN: 0.0, MAX: 100.0, STEP: 0.5}),
                "cfg_mode": (DynThresh.Modes, ),
                "cfg_scale_min": (FLOAT, {DEFAULT: 0.0, MIN: 0.0, MAX: 100.0, STEP: 0.5}),
                "sched_val": (FLOAT, {DEFAULT: 1.0, MIN: 0.0, MAX: 100.0, STEP: 0.01}),
                "separate_feature_channels": ([ENABLE, "disable"], ),
                "scaling_startpoint": (DynThresh.Startpoints, ),
                "variability_measure": (DynThresh.Variabilities, ),
                "interpolate_phi": (FLOAT, {DEFAULT: 1.0, MIN: 0.0, MAX: 1.0, STEP: 0.01}),
                }
        }

    RETURN_TYPES = (MODEL,)
    FUNCTION = "patch"
    CATEGORY = "advanced/mcmonkey"

    def patch(self, model, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi):

        dynamic_thresh = DynThresh(mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, 0, 999, separate_feature_channels == ENABLE, scaling_startpoint, variability_measure, interpolate_phi)
        
        def sampler_dyn_thresh(args):
            input = args["input"]
            cond = input - args["cond"]
            uncond = input - args["uncond"]
            cond_scale = args["cond_scale"]
            time_step = model.model.model_sampling.timestep(args["sigma"])
            time_step = time_step[0].item()
            dynamic_thresh.step = 999 - time_step

            return input - dynamic_thresh.dynthresh(cond, uncond, cond_scale, None)

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_dyn_thresh)
        return (m, )
