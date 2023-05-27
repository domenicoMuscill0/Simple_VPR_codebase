import torch
from torch import nn


class CRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(size=(20, 20), mode='bilinear', align_corners=True)
        self.multi_scale_filters = [
            nn.Conv2d(in_channels=512, out_channels=16, kernel_size=(8, 8), stride=2, padding=0),  # Templates
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(3, 3), stride=3, padding=1),  # Small
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(7, 7), stride=5, padding=9),  # Medium
            nn.Conv2d(in_channels=512, out_channels=20, kernel_size=(9, 9), stride=2, padding=1)  # Large
        ]
        self.accumulation = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x, t):
        x = self.upsample(x)                    # Upsample
        t = self.multi_scale_filters[1](t)      # Multiscale Context Filters
        xs = self.multi_scale_filters[1](x)
        xm = self.multi_scale_filters[2](x)
        xl = self.multi_scale_filters[3](x)
        x = torch.Stack([t, xs, xm, xl])        # Concatenation of 100 feature maps
        x = self.accumulation(x)                # Accumulation
        return x
