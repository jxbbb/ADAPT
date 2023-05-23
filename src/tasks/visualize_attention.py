from __future__ import absolute_import, division, print_function
import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import numpy as np
from PIL import Image
import os.path as op
import json
import time
import torch
import torch.distributed as dist
from apex import amp
import deepspeed
from src.configs.config import (basic_check_arguments, shared_configs)
from src.datasets.data_utils.video_ops import extract_frames_from_video_path
from src.datasets.data_utils.video_transforms import Compose, Resize, Normalize, CenterCrop
from src.datasets.data_utils.volume_transforms import ClipToTensor
from src.datasets.caption_tensorizer import build_tensorizer
from src.utils.deepspeed import fp32_to_fp16
from src.utils.logger import LOGGER as logger
from src.utils.logger import (TB_LOGGER, RunningMeter, add_log_to_file)
from src.utils.comm import (is_main_process,
                            get_rank, get_world_size, dist_init)
from src.utils.miscellaneous import (mkdir, set_seed, str_to_bool)
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.modeling.multitask_e2e_vid_swin_bert import MultitaskVideoTransformer
from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model
from PIL import Image
import numpy as np
# import pyttsx3

import cv2
# grad cam
from src.pytorch_grad_cam.grad_cam import GradCAM
from src.pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
import ffmpeg

def _online_video_decode(args, video_path):
    decoder_num_frames = getattr(args, 'max_num_frames', 2)
    frames, _ = extract_frames_from_video_path(
                video_path, target_fps=3, num_frames=decoder_num_frames,
                multi_thread_decode=False, sampling_strategy="uniform",
                safeguard_duration=False, start=3, end=10)

    rotate = ffmpeg.probe(video_path)['streams'][0]['tags']['rotate']
    rotate = int(rotate)
    if rotate != 0:
        print(f"Video is rotated, rotating...")
        new_frames = []
        for i in range(frames.shape[0]):
            frame = frames[i].permute(1,2,0).numpy()
            for _ in range((360-rotate)//90):
                frame = np.rot90(frame)
            new_frames.append(frame)
        print(f"Rotating done...")
        return torch.tensor(new_frames).permute(0, 3, 1, 2)
    return frames

def _transforms(args, frames):
    raw_video_crop_list = [
        Resize(args.img_res),
        CenterCrop((args.img_res,args.img_res)),
        ClipToTensor(channel_nb=3),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]
    raw_video_prcoess = Compose(raw_video_crop_list)

    frames = frames.numpy()
    frames = np.transpose(frames, (0, 2, 3, 1))
    num_of_frames, height, width, channels = frames.shape

    frame_list = []
    for i in range(args.max_num_frames):
        frame_list.append(Image.fromarray(frames[i]))

    # apply normalization, output tensor (C x T x H x W) in the range [0, 1.0]
    crop_frames = raw_video_prcoess(frame_list)
    # (C x T x H x W) --> (T x C x H x W)
    crop_frames = crop_frames.permute(1, 0, 2, 3)
    return crop_frames 

def reshape_transform(tensor, frame=16, height=7, width=7):
    result = tensor.reshape(tensor.size(0), frame,
                            height, width, tensor.size(-1))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(0, 4, 1, 2, 3)
    return result

def visualize_attn(args, video_path, model, tokenizer, tensorizer):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])

    model.float()
    model.eval()

    frames = _online_video_decode(args, video_path)
    preproc_frames = _transforms(args, frames)
    data_sample = tensorizer.tensorize_example_e2e("", preproc_frames, text_b="")
    data_sample = tuple(t.to(args.device) for t in data_sample)


    cam = GradCAM(model=model,
                  target_layers=[model.swin.backbone.norm],
                  use_cuda=True,
                  reshape_transform=reshape_transform
                )

    cur_len = 1
    sentence_id = 0
    while cur_len <= args.max_seq_length:


        masked_pos = torch.zeros_like(data_sample[0][None,:])
        masked_pos[:, cur_len] = 1

        inputs = {
            'input_ids': data_sample[0][None,:], 'attention_mask': data_sample[1][None,:],
            'token_type_ids': data_sample[2][None,:], 'img_feats': data_sample[3][None,:],
            'masked_pos': masked_pos,
            'masked_ids': torch.zeros((1), dtype=torch.int64),   # targets in grad cam, don't know before feeding the data, set it zero
        }
        # inputs = {'is_decode': True,
        #     'input_ids': data_sample[0][None,:], 'attention_mask': data_sample[1][None,:],
        #     'token_type_ids': data_sample[2][None,:], 'img_feats': data_sample[3][None,:],
        #     'masked_pos': data_sample[4][None,:],
        #     'do_sample': False,
        #     'bos_token_id': cls_token_id,
        #     'pad_token_id': pad_token_id,
        #     'eos_token_ids': [sep_token_id],
        #     'mask_token_id': mask_token_id,
        #     # for adding od labels
        #     'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
        #     # hyperparameters of beam search
        #     'max_length': args.max_gen_length if not args.use_sep_cap else args.max_gen_length*2,
        #     'use_sep_cap': args.use_sep_cap,
        #     'num_beams': args.num_beams,
        #     "temperature": args.temperature,
        #     "top_k": args.top_k,
        #     "top_p": args.top_p,
        #     "repetition_penalty": args.repetition_penalty,
        #     "length_penalty": args.length_penalty,
        #     "num_return_sequences": args.num_return_sequences,
        #     "num_keep_best": args.num_keep_best,
        # }
        tic = time.time()

        grayscale_cam = cam(input_tensor=inputs,
                        targets=None,
                        eigen_smooth=False,
                        aug_smooth=False)

        time_meter = time.time() - tic
        print(time_meter)

        # Here grayscale_cam has only one video in the batch
        grayscale_cam = grayscale_cam[0, :]

        cropped_raw_frames = CenterCrop((720,720))([frame.permute(1,2,0).numpy() for frame in frames])

        save_path = "grad_cam_vis_1"
        os.makedirs(save_path, exist_ok=True)

        output_word = cam.output_word.item()

        for i in range(len(grayscale_cam)):
            cam_img = grayscale_cam[i]
            cam_img = cv2.resize(cam_img, (720, 720))
            frame_id = 2*i+1
            raw_img = cropped_raw_frames[frame_id][:, :, ::-1] / 255

            cam_image = show_cam_on_image(raw_img, cam_img)
            cv2.imwrite(os.path.join(save_path, tokenizer.convert_ids_to_tokens(output_word)+str(i).zfill(3)+".jpg"), cam_image)


        if sentence_id == 0:
            if output_word == 102:
                sentence_id += 1
                data_sample[0][cur_len] = output_word
                cur_len += 1
                while cur_len < args.max_seq_a_length:
                    data_sample[0][cur_len] = 0
                    cur_len += 1
                data_sample[0][cur_len] = 101
                cur_len += 1
            else:
                data_sample[0][cur_len] = output_word
                cur_len += 1
        else:
            if output_word == 102:
                break
            else:
                data_sample[0][cur_len] = output_word
                cur_len += 1

    print(data_sample[0])
    print(tokenizer.convert_ids_to_tokens(data_sample[0].tolist()))

def check_arguments(args):
    # shared basic checks
    basic_check_arguments(args)
    # additional sanity check:
    args.max_img_seq_length = int((args.max_num_frames/2)*(int(args.img_res)/32)*(int(args.img_res)/32))
    
    if args.freeze_backbone or args.backbone_coef_lr == 0:
        args.backbone_coef_lr = 0
        args.freeze_backbone = True
    
    if 'reload_pretrained_swin' not in args.keys():
        args.reload_pretrained_swin = False

    if not len(args.pretrained_checkpoint) and args.reload_pretrained_swin:
        logger.info("No pretrained_checkpoint to be loaded, disable --reload_pretrained_swin")
        args.reload_pretrained_swin = False

    if args.learn_mask_enabled==True and args.attn_mask_type != 'learn_without_crossattn' and args.attn_mask_type != 'learn_with_swap_crossattn': 
        args.attn_mask_type = 'learn_vid_att'

def update_existing_config_for_inference(args):
    ''' load swinbert args for evaluation and inference 
    '''
    assert args.do_test or args.do_eval
    checkpoint = args.eval_model_dir
    try:
        json_path = op.join(checkpoint, os.pardir, 'log', 'args.json')
        f = open(json_path,'r')
        json_data = json.load(f)

        from easydict import EasyDict
        train_args = EasyDict(json_data)
    except Exception as e:
        train_args = torch.load(op.join(checkpoint, 'training_args.bin'))

    train_args.eval_model_dir = args.eval_model_dir
    train_args.resume_checkpoint = args.eval_model_dir + 'model.bin'
    train_args.model_name_or_path = 'bert-base-uncased'
    train_args.do_train = False
    train_args.do_eval = True
    train_args.do_signal_eval = True if hasattr(args, 'do_signal_eval') and args.do_signal_eval else False
    train_args.do_test = True
    train_args.val_yaml = args.val_yaml
    train_args.test_video_fname = args.test_video_fname
    train_args.use_car_sensor = True if hasattr(args, 'use_car_sensor') and args.use_car_sensor else False
    train_args.multitask = True if hasattr(args, 'multitask') and args.multitask else False
    return train_args

def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument("--grid_feat", type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument("--kinetics", type=str, default='400', help="400 or 600")
    parser.add_argument("--pretrained_2d", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--vidswin_size", type=str, default='base')
    parser.add_argument('--freeze_backbone', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_checkpoint', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument("--reload_pretrained_swin", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--learn_mask_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--loss_sensor_w', type=float, default=0)
    parser.add_argument('--sparse_mask_soft2hard', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--transfer_method', type=int, default=-1,
                        help="0: load all SwinBERT pre-trained weights, 1: load only pre-trained sparse mask")
    parser.add_argument('--att_mask_expansion', type=int, default=-1,
                        help="-1: random init, 0: random init and then diag-based copy, 1: interpolation")
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--test_video_fname', type=str, default='None')
    args = base_config.parse_args()
    return args

def main(args):
    args = update_existing_config_for_inference(args)
    # global training_saver
    args.device = torch.device(args.device)
    # Setup CUDA, GPU & distributed training
    dist_init(args)
    check_arguments(args)
    set_seed(args.seed, 0)
    fp16_trainning = None
    logger.info(
        "device: {}, n_gpu: {}, rank: {}, "
        "16-bits training: {}".format(
            args.device, args.num_gpus, get_rank(), fp16_trainning))

    if not is_main_process():
        logger.disabled = True

    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}" )

     # Get Video Swin model 
    swin_model = get_swin_model(args)
    # Get BERT and tokenizer 
    bert_model, config, tokenizer = get_bert_model(args)

    # build ADAPT based on training configs
    if args.multitask:
        vl_transformer = MultitaskVideoTransformer(args, config, swin_model, bert_model)
    else:
        vl_transformer = VideoTransformer(args, config, swin_model, bert_model)
    vl_transformer.freeze_backbone(freeze=args.freeze_backbone)

    # load weights for inference
    logger.info(f"Loading state dict from checkpoint {args.resume_checkpoint}")
    cpu_device = torch.device('cpu')
    pretrained_model = torch.load(args.resume_checkpoint, map_location=cpu_device)

    if isinstance(pretrained_model, dict):
        vl_transformer.load_state_dict(pretrained_model, strict=False)
    else:
        vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

    vl_transformer.to(args.device)
    vl_transformer.eval()

    tensorizer = build_tensorizer(args, tokenizer, is_train=False)
    visualize_attn(args, args.test_video_fname, vl_transformer, tokenizer, tensorizer)

if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)
