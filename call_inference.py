"""
use open-source lib to achieve analog inference
"""
import torch
# Import memtorch----------------------------------------------
# import memtorch
# import copy
# # Prior to conversion, a memristive device model must be defined and characterized in part
# from memtorch.mn.Module import patch_model #be used to convert DNNs to a MDNNs.
# from memtorch.map.Input import naive_scale
# from memtorch.map.Parameter import naive_map # used to convert the weights within all torch.nn.Conv2d layers to equivalent conductance values
# from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
#--------------------------------------------------------------
#Import aihwkit-----------------------------------------------
# from aihwkit.optim import AnalogSGD
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import TikiTakaEcRamPreset
from aihwkit.simulator.configs import MappingParameter, SingleRPUConfig, IOParameters, InferenceRPUConfig
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.simulator.configs import WeightModifierType, WeightNoiseType, WeightClipType
# from aihwkit.simulator.rpu_base import cuda
#--------------------------------------------------------------
# class infer_memtorch():
#     def __init__(self, r_on=1.4e4, r_off=5e7, tile_shape=(128, 128), ADC_resolution=8, failure_percentage=0.25) -> None:
#         self.r_on=r_on
#         self.r_off=r_off
#         self.tile_shape=tile_shape
#         self.ADC_resolution=ADC_resolution
#         self.failure_percentage=failure_percentage
#     def patch(self, model):
#         reference_memristor = memtorch.bh.memristor.VTEAM # A reference (base) memristor model
#         reference_memristor_params = {'time_series_resolution': 1e-10,'r_off': self.r_off, 'r_on': self.r_on}
#         memristor = reference_memristor(**reference_memristor_params)
#         patched_model = patch_model(copy.deepcopy(model),
#                           memristor_model=reference_memristor,
#                           memristor_model_params=reference_memristor_params,
#                           module_parameters_to_patch=[torch.nn.Linear, torch.nn.Conv2d],
#                           mapping_routine=naive_map,
#                           transistor=True,
#                           programming_routine=None,
#                           tile_shape=self.tile_shape,
#                           max_input_voltage=0.3,
#                           scaling_routine=naive_scale,
#                           ADC_resolution=self.ADC_resolution,
#                           ADC_overflow_rate=0.,
#                           quant_method='linear')
#         patched_model_ = apply_nonidealities(copy.deepcopy(patched_model),
#                                   non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
#                                   lrs_proportion=0.25,
#                                   hrs_proportion=0.10,
#                                   electroform_proportion=0)
#         patched_model_.tune_()
#         return patched_model_
    
class infer_aihwkit():
    def __init__(self, weight_scaling_omega=0.6, forward_out_res=-1., forward_w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT,
                  forward_w_noise=0.01, clip_type=WeightClipType.FIXED_VALUE, clip_fixed_value=1.0, modifier_pdrop=0.03, 
                  modifier_type=WeightModifierType.ADD_NORMAL, modifier_std_dev=0.1, modifier_rel_to_actual_wmax=True,
                  g_max=25.0, device='PCM', optional=None) -> None:
        
        self.forward_out_res = forward_out_res
        self.forward_w_noise_type = forward_w_noise_type  # Turn off (output) ADC discretization.
        self.forward_w_noise = forward_w_noise  # Short-term w-noise.
        self.weight_scaling_omega = weight_scaling_omega       
        self.clip_type = clip_type 
        self.clip_fixed_value = clip_fixed_value
        self.modifier_pdrop = modifier_pdrop
        self.modifier_type = modifier_type
        self.modifier_std_dev = modifier_std_dev
        self.modifier_rel_to_actual_wmax = modifier_rel_to_actual_wmax
        self.g_max = g_max
        self.optional = optional
        self.device = device

    def create_rpu_config(self):
        if self.device=='ReRAM':   
            from aihwkit.simulator.presets import ReRamSBPreset
            rpu_config = ReRamSBPreset() 
        elif self.device=='EcRAM':
            mapping = MappingParameter(self.weight_scaling_omega)
            rpu_config = TikiTakaEcRamPreset(mapping=mapping)
        else:
            rpu_config = InferenceRPUConfig()
            rpu_config.forward.out_res = self.forward_out_res  # Turn off (output) ADC discretization.
            rpu_config.forward.w_noise_type = self.forward_w_noise_type
            rpu_config.forward.w_noise = self.forward_w_noise  # Short-term w-noise.
            #---------------------------optional------------------------------------
            if self.optional:    
                rpu_config.clip.type = self.clip_type
                rpu_config.clip.fixed_value = self.clip_fixed_value
                rpu_config.modifier.pdrop = self.modifier_pdrop  # Drop connect.
                rpu_config.modifier.type = self.modifier_type  # Fwd/bwd weight noise.
                rpu_config.modifier.std_dev = self.modifier_std_dev
                rpu_config.modifier.rel_to_actual_wmax = self.modifier_rel_to_actual_wmax
            rpu_config.noise_model = PCMLikeNoiseModel(self.g_max)

        return rpu_config   

    def patch(self, model):
        RPU_CONFIG = self.create_rpu_config()
        analog_model = convert_to_analog(model, RPU_CONFIG)
        return analog_model


# class infer_MNSIM():
#     def __init__(self) -> None:
#         pass