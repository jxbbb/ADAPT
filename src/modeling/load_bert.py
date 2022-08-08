from src.layers.bert import BertTokenizer, BertConfig, BertForImageCaptioning
from src.utils.logger import LOGGER as logger

def get_bert_model(args):
    # Load pretrained bert and tokenizer based on training configs
    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=2, finetuning_task='image_captioning')

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
    config.img_feature_type = 'frcnn'
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = 'classification'
    config.tie_weights = args.tie_weights
    config.freeze_embedding = args.freeze_embedding
    config.label_smoothing = args.label_smoothing
    config.drop_worst_ratio = args.drop_worst_ratio
    config.drop_worst_after = args.drop_worst_after
    # update model structure if specified in arguments
    update_params = ['img_feature_dim', 'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
    model_structure_changed = [False] * len(update_params)
    # model_structure_changed[0] = True  # cclin hack
    for idx, param in enumerate(update_params):
        arg_param = getattr(args, param)
        # bert-base-uncased do not have img_feature_dim
        config_param = getattr(config, param) if hasattr(config, param) else -1
        if arg_param > 0 and arg_param != config_param:
            logger.info(f"Update config parameter {param}: {config_param} -> {arg_param}")
            setattr(config, param, arg_param)
            model_structure_changed[idx] = True
    if any(model_structure_changed):
        assert config.hidden_size % config.num_attention_heads == 0
        if args.load_partial_weights:
            # can load partial weights when changing layer only.
            assert not any(model_structure_changed[2:]), "Cannot load partial weights " \
                "when any of ({}) is changed.".format(', '.join(update_params[2:]))
            model = model_class.from_pretrained(args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
            logger.info("Load partial weights for bert layers.")
        else:
            model = model_class(config=config) # init from scratch
            logger.info("Init model from scratch.")
    else:
        model = model_class.from_pretrained(args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        logger.info(f"Load pretrained model: {args.model_name_or_path}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model total parameters: {total_params}')
    return model, config, tokenizer