"""
Modified from ClipBERT code
"""
import os
import sys
import json
import argparse
import torch

from easydict import EasyDict as edict
from src.utils.miscellaneous import str_to_bool, check_yaml_file
from src.utils.logger import LOGGER
from os import path as op
from packaging import version


def parse_with_config(parsed_args):
    """This function will set args based on the input config file.
    (1) it only overwrites unset parameters,
        i.e., these parameters not set from user command line input
    (2) it also sets configs in the config file but declared in the parser
    """
    # convert to EasyDict object, enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                         if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


class SharedConfigs(object):
    """Shared options for pre-training and downstream tasks.
    For each downstream task, implement a get_*_args function,
    see `get_pretraining_args()`

    Usage:
    >>> shared_configs = SharedConfigs()
    >>> pretraining_config = shared_configs.get_pretraining_args()
    """

    def __init__(self, desc="shared config"):
        parser = argparse.ArgumentParser(description=desc)
        # path configs
        parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                            help="Directory with all datasets, each in one subfolder")
        parser.add_argument("--output_dir", default='output/', type=str, required=False,
                            help="The output directory to save checkpoint and test results.")
        parser.add_argument("--train_yaml", default='coco_caption/train.yaml', type=str, required=False,
                            help="Yaml file with all data for training.")

        # multimodal transformer modeling config
        parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                            help="Path to pre-trained model or model type.")
        parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name.")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name.")
        parser.add_argument("--num_hidden_layers", default=-1, type=int, required=False,
                            help="Update model config if given")
        parser.add_argument("--hidden_size", default=-1, type=int, required=False,
                            help="Update model config if given")
        parser.add_argument("--num_attention_heads", default=-1, type=int, required=False,
                            help="Update model config if given. Note that the division of "
                            "hidden_size / num_attention_heads should be in integer.")
        parser.add_argument("--intermediate_size", default=-1, type=int, required=False,
                            help="Update model config if given.")
        parser.add_argument("--img_feature_dim", default=512, type=int,
                            help="Update model config if given.The Image Feature Dimension.")
        parser.add_argument("--load_partial_weights", type=str_to_bool, nargs='?',
                            const=True, default=False,
                            help="Only valid when change num_hidden_layers, img_feature_dim, but not other structures. "
                            "If set to true, will load the first few layers weight from pretrained model.")
        parser.add_argument("--freeze_embedding", type=str_to_bool, nargs='?', const=True, default=False,
                            help="Whether to freeze word embeddings in Bert")
        parser.add_argument("--drop_out", default=0.1, type=float,
                            help="Drop out ratio in BERT.")

        # inputs to multimodal transformer config
        parser.add_argument("--max_seq_length", default=70, type=int,
                            help="The maximum total input sequence length after tokenization.")
        parser.add_argument("--max_seq_a_length", default=40, type=int,
                            help="The maximum sequence length for caption.")
        parser.add_argument("--max_img_seq_length", default=50, type=int,
                            help="The maximum total input image sequence length.")
        parser.add_argument("--do_lower_case", type=str_to_bool, nargs='?', const=True, default=False,
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--add_od_labels", type=str_to_bool, nargs='?', const=True, default=False,
                            help="Whether to add object detection labels or not")
        parser.add_argument("--od_label_conf", default=0.0, type=float,
                            help="Confidence threshold to select od labels.")
        parser.add_argument("--use_asr", type=str_to_bool, nargs='?', const=True, default=False,
                            help="Whether to add ASR/transcript as additional modality input")
        parser.add_argument("--unique_labels_on", type=str_to_bool, nargs='?', const=True, default=False,
                            help="Use unique labels only.")
        parser.add_argument("--no_sort_by_conf", type=str_to_bool, nargs='?', const=True, default=False,
                            help="By default, we will sort feature/labels by confidence, "
                            "which is helpful when truncate the feature/labels.")
        #======= mask token
        parser.add_argument("--mask_prob", default=0.15, type=float,
                            help= "Probability to mask input sentence during training.")
        parser.add_argument("--max_masked_tokens", type=int, default=3,
                            help="The max number of masked tokens per sentence.")
        parser.add_argument("--attn_mask_type", type=str, default='seq2seq',
                           choices=['seq2seq', 'bidirectional', 'learn_vid_mask'], 
                            help="Attention mask type, support seq2seq, bidirectional")
        parser.add_argument("--text_mask_type", type=str, default='random',
                           choices=['random', 'pos_tag', 'bert_attn', 'attn_on_the_fly'], 
                            help="Attention mask type, support random, pos_tag, bert_attn (precomputed_bert_attn), attn_on_the_fly")
        parser.add_argument("--tag_to_mask", default=["noun", "verb"], type=str, nargs="+", 
                            choices=["noun", "verb", "adjective", "adverb", "number"],
                            help= "what tags to mask")
        parser.add_argument("--mask_tag_prob", default=0.8, type=float,
                            help= "Probability to mask input text tokens with included tags during training.")
        parser.add_argument("--tagger_model_path", type=str, default='models/flair/en-pos-ontonotes-fast-v0.5.pt', 
                            help="checkpoint path to tagger model")
        parser.add_argument("--random_mask_prob", default=0, type=float,
                            help= "Probability to mask input text tokens randomly when using other text_mask_type")

        # data loading
        parser.add_argument("--on_memory", type=str_to_bool, nargs='?', const=True, default=False,
                            help="Option to load labels/caption to memory before training.")
        parser.add_argument("--effective_batch_size", default=-1, type=int,
                            help="Batch size over all GPUs for training.")
        parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--num_workers", default=4, type=int,
                            help="Workers in dataloader.")
        parser.add_argument('--limited_samples', type=int, default=-1, 
                            help="Set # of samples per node. Data partition for cross-node training.")
        
        # training configs
        parser.add_argument("--learning_rate", default=3e-5, type=float,
                            help="The initial lr.")
        parser.add_argument("--weight_decay", default=0.05, type=float,
                            help="Weight deay.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--warmup_ratio", default=0.1, type=float,
                            help="Linear warmup.")
        parser.add_argument("--scheduler", default='warmup_linear', type=str,
                            help="warmup_linear (triangle) or step",
                            choices=["warmup_linear", "step"])
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
        parser.add_argument("--num_train_epochs", default=20, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument('--logging_steps', type=int, default=20,
                            help="Log every X steps.")
        parser.add_argument('--save_steps', type=int, default=2000,
                            help="Save checkpoint every X steps. Will also perform evaluatin.")
        parser.add_argument('--restore_ratio', type=float, default=0.05,
                            help="save restorer checkpoint for 0.05 ratio")
        parser.add_argument("--device", type=str, default='cuda',
                            help="cuda or cpu")
        parser.add_argument('--seed', type=int, default=88,
                            help="random seed for initialization.")
        parser.add_argument("--local_rank", type=int, default=0,
                            help="For distributed training.")
        # ========= mix-precision training (>torch1.6 only)
        parser.add_argument('--mixed_precision_method', default='apex', type=str,
                            help="set mixed_precision_method, options: apex, deepspeed, fairscale",
                            choices=["apex", "deepspeed", "fairscale"])
        parser.add_argument('--zero_opt_stage', type=int,
                            help="zero_opt_stage, only allowed in deepspeed", 
                            default=-1, choices=[0, 1, 2, 3])
        parser.add_argument('--amp_opt_level', default=0,
                            help="amp optimization level, can set for both deepspeed and apex",  type=int,
                            choices=[0, 1, 2, 3])
        parser.add_argument('--deepspeed_fp16',
                            help="use fp16 for deepspeed",  type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--fairscale_fp16',
                            help="use fp16 for fairscale",  type=str_to_bool, nargs='?', const=True, default=False)
        # ========= resume training or load pre_trained weights
        parser.add_argument('--pretrained_checkpoint', type=str, default='')

        # for debug purpose
        parser.add_argument('--debug', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--debug_speed', type=str_to_bool, nargs='?', const=True, default=False)

        # can use config files, will only overwrite unset parameters
        parser.add_argument("--config", help="JSON config files")
        self.parser = parser

    def parse_args(self):
        parsed_args = self.parser.parse_args()
        args = parse_with_config(parsed_args)
        return args

    def add_downstream_args(self):
        # downstream finetuning args (not needed for pretraining)
        self.parser.add_argument("--eval_model_dir", type=str, default='',
                                 help="Model directory for evaluation.")
        
        # training/validation/inference mode (only needed for captioning)
        self.parser.add_argument("--val_yaml", default='coco_caption/val.yaml',
                                 type=str, required=False,
                                 help="Yaml file with all data for validation")
        self.parser.add_argument("--test_yaml", default='coco_caption/test.yaml', type=str,
                                 required=False, nargs='+',
                                 help="Yaml file with all data for testing, could be multiple files.")

        self.parser.add_argument("--do_train", type=str_to_bool, nargs='?',
                                 const=True, default=False,
                                 help="Whether to run training.")
        self.parser.add_argument("--do_test", type=str_to_bool,
                                 nargs='?', const=True, default=False,
                                 help="Whether to run inference.")
        self.parser.add_argument("--do_eval", type=str_to_bool, nargs='?',
                                 const=True, default=False,
                                 help="Whether to run evaluation.")
        self.parser.add_argument("--evaluate_during_training", type=str_to_bool,
                                 nargs='?', const=True, default=False,
                                 help="Run evaluation during training at each save_steps.")
        self.parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                                 help="Batch size per GPU/CPU for evaluation.")
        return

    def shared_video_captioning_config(self, cbs=False, scst=False):
        self.add_downstream_args()
        # image feature masking (only used in captioning?)
        self.parser.add_argument('--mask_img_feat', type=str_to_bool,
                                 nargs='?', const=True, default=False,
                                 help='Enable image fetuare masking')
        self.parser.add_argument('--max_masked_img_tokens', type=int, default=10,
                                 help="Maximum masked object featrues")

        # basic decoding configs
        self.parser.add_argument("--tie_weights", type=str_to_bool, nargs='?',
                                 const=True, default=False,
                                 help="Whether to tie decoding weights to that of encoding")
        self.parser.add_argument("--label_smoothing", default=0, type=float,
                                 help=".")
        self.parser.add_argument("--drop_worst_ratio", default=0, type=float,
                                 help=".")
        self.parser.add_argument("--drop_worst_after", default=0, type=int,
                                 help=".")
        self.parser.add_argument('--max_gen_length', type=int, default=20,
                                 help="max length of generated sentences")
        self.parser.add_argument('--output_hidden_states', type=str_to_bool,
                                 nargs='?', const=True, default=False,
                                 help="Turn on for fast decoding")
        self.parser.add_argument('--num_return_sequences', type=int, default=1,
                                 help="repeating times per image")
        self.parser.add_argument('--num_beams', type=int, default=1,
                                 help="beam search width")
        self.parser.add_argument('--num_keep_best', type=int, default=1,
                                 help="number of hypotheses to keep in beam search")
        self.parser.add_argument('--temperature', type=float, default=1,
                                 help="temperature in softmax for sampling")
        self.parser.add_argument('--top_k', type=int, default=0,
                                 help="filter distribution for sampling")
        self.parser.add_argument('--top_p', type=float, default=1,
                                 help="filter distribution for sampling")
        self.parser.add_argument('--repetition_penalty', type=int, default=1,
                                 help="repetition penalty from CTRL paper "
                                 "(https://arxiv.org/abs/1909.05858)")
        self.parser.add_argument('--length_penalty', type=int, default=1,
                                 help="beam search length penalty")
        
        if cbs:
            self.constraint_beam_search_args()
        if scst:
            self.self_critic_args()

        return
    
    def constraint_beam_search_args(self):
        
        # for Constrained Beam Search
        self.parser.add_argument('--use_cbs', type=str_to_bool, nargs='?', const=True, default=False,
                                 help='Use constrained beam search for decoding')
        self.parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                                 help="minimum number of constraints to satisfy")
        self.parser.add_argument('--use_hypo', type=str_to_bool, nargs='?', const=True, default=False,
                                 help='Store hypotheses for constrained beam search')
        self.parser.add_argument('--decoding_constraint', type=str_to_bool, nargs='?',
                                 const=True, default=False,
                                 help='When decoding enforce the constraint that the'
                                 'word cannot be consecutively predicted twice in a row')
        self.parser.add_argument('--remove_bad_endings', type=str_to_bool, nargs='?',
                                 const=True, default=False,
                                 help='When decoding enforce that the tokens in bad endings,'
                                 'e.g., a, the, etc cannot be predicted at the end of the sentence')
        return

    def self_critic_args(self):
        # for self-critical sequence training
        self.parser.add_argument('--scst', type=str_to_bool, nargs='?', const=True, default=False,
                                 help='Self-critical sequence training')
        self.parser.add_argument('--sc_train_sample_n', type=int, default=5,
                                 help="number of sampled captions for sc training")
        self.parser.add_argument('--sc_baseline_type', type=str, default='greedy',
                                 help="baseline tyep of REINFORCE algorithm")
        self.parser.add_argument('--cider_cached_tokens', type=str,
                                 default='coco_caption/gt/coco-train-words.p',
                                 help="path to cached cPickle file used to calculate CIDEr scores")
        return

shared_configs = SharedConfigs()

def basic_check_arguments(args):
    args.output_dir = args.output_dir.replace(" ", "_")
    if args.debug_speed:
        args.logging_steps = 1
        args.num_train_epochs = 1

    if args.debug:
        args.effective_batch_size = args.num_gpus
        args.per_gpu_train_batch_size = 1
        args.num_train_epochs = 1
        args.logging_steps = 5
        args.max_img_seq_length = 98

    # can add some basic checks here
    if args.mixed_precision_method != "deepspeed":
        LOGGER.info("Deepspeed is not enabled. We will disable the relevant args --zero_opt_stage and --deepspeed_fp16.")
        args.zero_opt_stage = -1
        args.deepspeed_fp16 = False
    
    if args.mixed_precision_method != "fairscale":
        LOGGER.info("Fairscale is not enabled. We will disable the relevant args --fairscale_fp16.")
        args.zero_opt_stage = -1
        args.fairscale_fp16 = False
    
    if args.mixed_precision_method != "apex":
        LOGGER.info("Disable restorer for deepspeed or fairscale")
        args.restore_ratio = -1
    
    if args.text_mask_type != "pos_tag":
        LOGGER.info("Disable --mask_tag_prob")
        args.mask_tag_prob = -1

    if hasattr(args, 'do_train') and args.do_train:
        check_yaml_file(op.join(args.data_dir, args.train_yaml))
        if args.evaluate_during_training:
            check_yaml_file(op.join(args.data_dir, args.val_yaml))
        # check after num_gpus initialized
        if args.effective_batch_size > 0:
            assert args.effective_batch_size % args.num_gpus == 0
            args.per_gpu_train_batch_size = int(args.effective_batch_size / args.num_gpus)
            args.per_gpu_eval_batch_size = int(args.effective_batch_size / args.num_gpus)
        else:
            assert args.per_gpu_train_batch_size > 0
            args.effective_batch_size = args.per_gpu_train_batch_size * args.num_gpus
            args.per_gpu_eval_batch_size = max(
                args.per_gpu_eval_batch_size, args.per_gpu_train_batch_size)

        if args.use_asr:
            args.add_od_labels = True
        if args.add_od_labels:
            assert args.max_seq_length > args.max_seq_a_length
        else:
            assert args.max_seq_length == args.max_seq_a_length
    if hasattr(args, 'do_test') and args.do_test:
        for test_yaml in args.test_yaml:
            check_yaml_file(op.join(args.data_dir, test_yaml))

def restore_training_settings(args):
    ''' Restore args for inference and SCST training
    Only works for downstream finetuning
    '''
    if args.do_train:
        if hasattr(args, 'scst') and not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    try:
        # train_args = torch.load(op.join(checkpoint, os.pardir, 'log', 'args.json')) #
        json_path = op.join(checkpoint, os.pardir, 'log', 'args.json')
        f = open(json_path,'r')
        json_data = json.load(f)
        from easydict import EasyDict
        train_args = EasyDict(json_data)
    except Exception as e:
        train_args = torch.load(op.join(checkpoint, 'training_args.bin'))

    if args.add_od_labels:
        if hasattr(train_args, 'max_seq_a_length'):
            if hasattr(train_args, 'scst') and train_args.scst:
                max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
            else:
                max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
            max_seq_length = args.max_gen_length + max_od_labels_len
            args.max_seq_length = max_seq_length
            LOGGER.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
                    max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['do_lower_case', 'add_od_labels',
            'img_feature_dim', 'no_sort_by_conf','num_hidden_layers']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                LOGGER.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)

    if hasattr(args, 'scst') and args.scst==True:
        args.max_seq_length = train_args.max_gen_length
        args.max_seq_a_length = train_args.max_gen_length
    return args

