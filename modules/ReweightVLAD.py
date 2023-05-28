import torch
from torch import nn
import torch.nn.functional as F


class CRN(nn.Module):
    def __init__(self):
        super(CRN, self).__init__()
        self.normalization = nn.LayerNorm([7, 7])
        self.upsample = nn.Upsample(size=(20, 20), mode='bilinear', align_corners=True)
        self.multi_scale_filters = [
            nn.Conv2d(in_channels=512, out_channels=16, kernel_size=(8, 8), stride=2, padding=0).half().cuda(),  # Templates
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(3, 3), stride=3, padding=1).half().cuda(),  # Small
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(7, 7), stride=5, padding=9).half().cuda(),  # Medium
            nn.Conv2d(in_channels=512, out_channels=20, kernel_size=(9, 9), stride=2, padding=1).half().cuda()  # Large
        ]
        self.accumulation = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x, t=None):
        # We apply Layer Normalization to enhance the presence of templates
        x = self.normalization(x)
        x = self.upsample(x)  # Upsample
        xs = self.multi_scale_filters[1](x)     # Multiscale Context Filters
        xm = self.multi_scale_filters[2](x)
        xl = self.multi_scale_filters[3](x)
        filters = [xs, xm, xl]
        if t is not None:
            t = self.upsample(t)
            t = self.multi_scale_filters[0](t)
            filters.append(t)
        x = torch.cat(filters, dim=1)  # Concatenation of 100 feature maps
        x = self.accumulation(x)  # Accumulation
        return x


class ReweightVLAD(nn.Module):
    """ReweightVLAD layer implementation"""

    def __init__(self, num_clusters=23, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters. By default, it is equal to the number of places
                in the dataset so that we can join it with GPM
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(ReweightVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.crn = CRN()
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x, t=None):
        N, C = x.shape[:2]

        # Generate the weights for context re-modulation
        context_weights = self.crn.forward(x, t).view(N, 1, -1)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)

        # Reweighting
        reweighting_mask = soft_assign * context_weights

        # VLAD core
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= reweighting_mask.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
