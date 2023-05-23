import torch
from fairscale.nn.misc import checkpoint_wrapper
import random


class VideoTransformer(torch.nn.Module):
    """ This is the one head module that performs Dirving Caption Generation. """
    def __init__(self, args, config, swin, transformer_encoder):
        super(VideoTransformer, self).__init__()
        """ Initializes the model.
        Parameters:
            args: basic args of ADAPT, mostly defined in `src/configs/VidSwinBert/BDDX_multi_default.json` and input args
            config: config of transformer_encoder, mostly defined in `models/captioning/bert-base-uncased/config.json`
            swin: torch module of the backbone to be used. See `src/modeling/load_swin.py`
            transformer_encoder: torch module of the transformer architecture. See `src/modeling/load_bert.py`
        """
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length

        self.max_num_frames = getattr(args, 'max_num_frames', 2)
        self.expand_car_info = torch.nn.Linear(self.max_num_frames, self.img_feature_dim)

        # add sensor information
        self.use_car_sensor = getattr(args, 'use_car_sensor', False)

        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        
        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        """ The forward process of ADAPT, 
        Parameters:
            input_ids: word tokens of input sentences tokenized by tokenizer
            attention_mask: multimodal attention mask in Vision-Language transformer
            token_type_ids: typen tokens of input sentences, 
                            0 means it is a narration sentence and 1 means a reasoning sentence, same size with input_ids
            img_feats: preprocessed frames of the video
            masked_pos: [MASK] position when performing MLM, used to locate the masked words
            masked_ids: groung truth of [MASK] when performing MLM
        """
        # grad cam can only input a tuple (args, kwargs)
        if isinstance(args, tuple) and len(args) != 0:
            kwargs = args[0]
            args= ()

        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        vid_feats = self.swin(images)

        # tokenize video features to video tokens
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)

        # use an mlp to transform video token dimension
        vid_feats = self.fc(vid_feats)

        # use video features to predict car tensor
        if self.use_car_sensor:
            car_infos = kwargs['car_info']
            car_infos = self.expand_car_info(car_infos)
            vid_feats = torch.cat((vid_feats, car_infos), dim=1)

        # prepare VL transformer inputs
        kwargs['img_feats'] = vid_feats

        # disable bert attention outputs to avoid some bugs
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)

        # learn soft attention mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask)*learn_att
            learn_att = diag_mask + video_attention
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att>=0.5)*1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att

        # Driving Caption Generation head
        outputs = self.trans_encoder(*args, **kwargs)

        # sparse attention mask loss
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)  
            outputs = outputs + (loss_sparsity, )

        return outputs
    
    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

 