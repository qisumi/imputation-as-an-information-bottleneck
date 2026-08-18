import torch
import torch.nn as nn
from .mask_layer import Mask_layer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin

        self.use_tq = True
        self.channel_aggre = True

        if self.use_tq:
            self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5)

        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, x, cycle_index):

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)

        if self.use_tq:
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=query_input, key=x_input, value=x_input)[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=x_input, key=x_input, value=x_input)[0]
            else:
                channel_information = 0

        input = self.input_proj(x_input+channel_information)

        hidden = self.model(input)

        output = self.output_proj(hidden+input).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output

class TQNetBackbone(nn.Module):

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        mask_layer_enable: bool = False,
        mask_kernel_size: int = 2,
        mask_dropout: float = 0.25,
        mask_backbone: str = "conv2d",
        d_model: int = 64,
        dropout: float = 0.1,
        use_revin: bool = True,
        cycle: int | None = None,
    ):
        super(TQNetBackbone, self).__init__()

        class Config:
            pass

        cfg = Config()
        cfg.seq_len = seq_len
        cfg.pred_len = pred_len
        cfg.enc_in = num_nodes
        cfg.cycle = cycle if cycle is not None else seq_len
        cfg.model_type = "TQNet"
        cfg.d_model = d_model
        cfg.dropout = dropout
        cfg.use_revin = use_revin

        self.tqnet = Model(cfg)
        self.mask_layer_enable = mask_layer_enable

        if self.mask_layer_enable:
            self.mask_layer = Mask_layer(
                feature_channel=1,
                kernel_size=mask_kernel_size,
                dropout=mask_dropout,
                backbone=mask_backbone,
            )

        self.receptive_field = seq_len

    def forward(self, input, idx=None):

        x = input[:, 0:1, :, :]

        if self.mask_layer_enable:
            mask = torch.isnan(x[:, 0, :, :]).type(torch.float32)
            expanded_mask = mask.unsqueeze(1)
            x = torch.cat((x, expanded_mask), dim=1)
            x[torch.isnan(x)] = 0
            x = self.mask_layer(x)
        else:
            x[torch.isnan(x)] = 0

        x = x.squeeze(1)
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        cycle_index = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        out = self.tqnet(x, cycle_index)
        out = out.unsqueeze(-1)
        return out
