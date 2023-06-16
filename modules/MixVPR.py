from torch import nn
import torch.nn.functional as f


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = []
        self.mix.append(nn.LayerNorm(in_dim))
        self.mix.append(nn.Linear(in_dim, int(in_dim * mlp_ratio)))
        self.mix.append(nn.ReLU())
        self.mix.append(nn.Linear(int(in_dim * mlp_ratio), in_dim))
        self.mix = nn.Sequential(*self.mix)

        # Ask for this snippet of code
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimension

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid-projection layer in the mixer block

        hw = in_h * in_w
        self.mix = []
        for _ in range(self.mix_depth):
            self.mix.append(FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio))
            hw = hw * mlp_ratio
        self.mix = nn.Sequential(*self.mix)
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        # _, _, V = torch.pca_lowrank(x)
        # x = matmul(x, V[:, :self.out_channels])
        x = self.mix(x)
        # What if we mix the feature maps by using skip connections?
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = f.normalize(x.flatten(1), p=2, dim=-1)
        return x
