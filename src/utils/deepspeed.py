from .logger import LOGGER as logger
from pprint import pformat
import torch


def get_deepspeed_config(args):
        config_params = {
            'train_batch_size': args.effective_batch_size,
        }

        use_fp16 = args.deepspeed_fp16
        use_amp = not args.deepspeed_fp16  # by default, if not use deepspeed fp16, will enable deepspeed amp 

        if use_amp:
            config_params['amp'] = {
                'enabled': True,
                'opt_level': f'O{args.amp_opt_level}',
            }

        if use_fp16:
            config_params['fp16'] = {
                'enabled': True,
            }

        gradient_clip = args.max_grad_norm
        if gradient_clip:
            config_params['gradient_clipping'] = gradient_clip

        config_params['flops_profiler'] = {
            'enabled': False,
            'profile_step': 1,
            'module_depth': -1,
            'top_modules': 3,
            'detailed': True,
        }

        config_params['logging'] = {
            'steps_per_print': args.logging_steps*10,
        }
        if hasattr(args, "zero_opt_stage") and args.zero_opt_stage > 0:
            config_params['zero_optimization'] = {
                'stage': args.zero_opt_stage,
            }
            if args.zero_opt_stage > 0:
                config_params['fp16'] = {
                    'enabled': True
                }
            config_params['zero_allow_untested_optimizer'] = True

        logger.info(pformat(config_params))
        return config_params


                
def fp32_to_fp16(inputs):
    # deepspeed does not auto cast inputs.
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            v = v.to(dtype=torch.half)
        inputs[k] = v
    return inputs
