import torch
import torch.nn as nn

class Mask_layer(nn.Module):

    def __init__(
        self,
        feature_channel: int = 1,
        kernel_size: int = 2,
        dropout: float = 0.25,
        backbone: str = "conv2d",
        mlp_hidden_dim: int = 128,
        attn_heads: int = 1,
    ):
        super(Mask_layer, self).__init__()
        self.feature_channel = feature_channel
        self.backbone = backbone.lower()
        self.dropout_p = dropout
        in_channels = feature_channel + 1

        if self.backbone == "conv2d":
            self.mask_layer = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=kernel_size),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(128, 512, kernel_size=kernel_size),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.ConvTranspose2d(512, 64, kernel_size=kernel_size),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.ConvTranspose2d(64, feature_channel, kernel_size=kernel_size),
            )
        elif self.backbone == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, feature_channel),
            )
        elif self.backbone == "attention":
            embed_dim = in_channels
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=False,
            )
            self.attn_proj = nn.Linear(embed_dim, feature_channel)
            self.attn_dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unsupported mask_layer backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone == "conv2d":
            return self.mask_layer(x)

        b, c, n, t = x.shape

        if self.backbone == "mlp":
            x_perm = x.permute(0, 2, 3, 1).contiguous()
            x_flat = x_perm.view(b * n * t, c)
            out_flat = self.mlp(x_flat)
            out = out_flat.view(b, n, t, self.feature_channel)
            out = out.permute(0, 3, 1, 2).contiguous()
            return out

        if self.backbone == "attention":

            x_seq = x.view(b, c, n * t).permute(2, 0, 1)
            attn_out, _ = self.attn(x_seq, x_seq, x_seq)
            attn_out = self.attn_dropout(attn_out)
            proj = self.attn_proj(attn_out)
            proj = proj.permute(1, 2, 0)
            out = proj.view(b, self.feature_channel, n, t)
            return out

        raise RuntimeError(f"Unexpected backbone type: {self.backbone}")

if __name__ == "__main__":
    for backbone in ["conv2d", "mlp", "attention"]:
        model = Mask_layer(feature_channel=1, kernel_size=2, backbone=backbone)
        data = torch.randn(size=(4, 2, 10, 12))
        res = model(data)
        print(backbone, res.shape)
