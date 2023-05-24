import torch
from src.utils.logger import LOGGER as logger
from torch import nn
from src.layers.bert.modeling_bert import BertEncoder
from src.layers.bert import BertConfig, BertEncoder

def get_sensor_pred_model(args):
    return Sensor_Pred_Head(args)


class Sensor_Pred_Head(torch.nn.Module):
    """ This is the Control Signal Prediction head that performs sensor regression """
    def __init__(self, args):
        """ Initializes the prediction head.
        A simple transformer that performs sensor regression. 
        We simply use a transformer to regress the whole signals of a video, which is superficial and could be optimized to a large extent.
        """
        super(Sensor_Pred_Head, self).__init__()

        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat

        # Motion Transformer implemented by bert
        self.config = BertConfig.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=2, finetuning_task='image_captioning')
        self.encoder = BertEncoder(self.config)

        # type number of control signals to be used
        # TODO: Set this variable as an argument, corresponging to the control signal in dataloader
        
        self.sensor_dim = len(args.signal_types)
        self.sensor_embedding = torch.nn.Linear(self.sensor_dim, self.config.hidden_size)
        self.sensor_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # a mlp to transform the dimension of video feature 
        self.img_dim = self.img_feature_dim
        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.img_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # a sample regression decoder
        self.decoder = nn.Linear(self.config.hidden_size, self.sensor_dim)


    def forward(self, *args, **kwargs):
        """The forward process.
        Parameters:
            img_feats: video features extracted by video swin
            car_info: ground truth of control signals
        """
        vid_feats = kwargs['img_feats']
        car_info  = kwargs['car_info']

        car_info = car_info.permute(0, 2, 1)

        B, S, C = car_info.shape
        assert C == self.sensor_dim, f"{C}, {self.sensor_dim}"
        frame_num = S

        img_embedding_output = self.img_embedding(vid_feats)
        img_embedding_output = self.img_dropout(img_embedding_output)


        extended_attention_mask = self.get_attn_mask(img_embedding_output)

        encoder_outputs = self.encoder(img_embedding_output,
                                        extended_attention_mask)
        sequence_output = encoder_outputs[0][:, :frame_num, :]

        pred_tensor = self.decoder(sequence_output)

        loss = self.get_l2_loss(pred_tensor, car_info)

        return loss, pred_tensor

    def get_attn_mask(self, img_embedding_output):
        """Get attention mask that should be passed to motion transformer."""
        device = img_embedding_output.device
        bsz = img_embedding_output.shape[0]
        img_len = img_embedding_output.shape[1]


        attention_mask = torch.ones((bsz, img_len, img_len), dtype=torch.long)


        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask.to(device)

    def get_l2_loss(self, pred, targ):
        loss_func = nn.MSELoss()
        return loss_func(pred, targ)