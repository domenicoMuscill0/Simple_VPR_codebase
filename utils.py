from collections import defaultdict
import faiss
import logging
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_metric_learning import losses
from matplotlib import pyplot as plt
from torch.utils.data import BatchSampler, Sampler
from torchvision.transforms.transforms import math
import visualizations
from datasets.test_dataset import TestDataset
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA

# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


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
            if isinstance(m, (nn.Linear)):
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
          hw = hw*mlp_ratio
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
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class ProxyHead(nn.Module):
    def __init__(self, out_dim: int = 32, in_dim: int = 128,
                 n_classes: int = 23, n_layers: int = 3):
        super().__init__()
        a, b = in_dim, -np.log(out_dim / in_dim) / n_layers
        dims = np.ceil(a * np.exp(-b * np.arange(n_layers + 1))).astype(np.int16)
        print("DIMENSIONI LINEAR", dims)
        self.reduction = [nn.Flatten()] + \
                     [nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU()) for i in range(n_layers)] + \
                     [nn.BatchNorm1d(out_dim)]
        self.reduction = nn.Sequential(*self.reduction)
        self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.001)


    def forward(self, images):
        descriptors = self.reduction(images)
        return descriptors

    def fit(self, descriptors, labels):
        compressed_descriptors = self(descriptors)
        loss = self.loss_fn(compressed_descriptors, labels)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

def proxy_factory(dim):
  return lambda: ProxyBank.Proxy(dim)

class ProxyBank:
    """This class stores the places' proxies together with their identifier
       and performs exhaustive search on the index to retrieve the mini-batch sampling pool."""

    def __init__(self, k: int = 10, dim: int =128, bank=None):
        self.__bank = defaultdict(proxy_factory(dim))
        self.k = k
        self.dim = dim
        self.nb_samples = self.dim
        self.index = faiss.index_factory(self.dim, 'IDMap,Flat')

    def update_bank(self, embs, labels):
        """This method adds descriptors and labels retrieved at each batch training step end to the underlying bank
           and the proxy index is updated."""

        for d, l in zip(embs, labels):
            self.__bank[l.item()] = self.__bank[l.item()] + ProxyBank.Proxy(d)
        idx = np.arange(self.nb_samples)
        embs_by_idx = torch.stack([self.__bank[i].get() for i in idx]).cpu()
        self.index.add_with_ids(embs_by_idx, idx)

    def build_index(self):
        """
        embs is the bank, labels are the place IDs, k is the number of places per group (M in the article)
        """

        embs = self.__bank.values()
        embs = list(map(lambda emb: emb.get().cpu(), embs))
        embs = np.vstack(embs)

        labels = torch.Tensor(list(self.__bank.keys()))

        s = {}
        for e in range(self.nb_samples):
            s[e] = 1

        ids = []
        list_idx = np.arange(self.nb_samples)
        np.random.shuffle(list_idx)
        for i in list_idx:
            if s[i] == 1:  # if id 'i' has not already been selected in one of the groups
                _, line = self.index.search(embs[i:i + 1], self.k)
                line = line[0]
                ids.extend(line)
                self.index.remove_ids(line)
                for e in line:
                    s[e] = 0  # keep track of the already selected ids

        ids = ids[:embs.shape[0]]
        return labels[ids]

    class Proxy:
        def __init__(self, tensor: torch.Tensor = None, n: int = 1, dim: int = 128):
            if tensor is None:
                self.__arch = torch.zeros(dim)
            else:
                self.__arch = tensor
            self.__n = n

        @property
        def get_shape(self):
            return self.__arch.shape

        def get(self):
            return self.__arch / self.__n

        def __add__(self, other):
            return ProxyBank.Proxy(tensor=self.__arch + other.__arch, n=self.__n + other.__n)


class HardSampler(Sampler):

    def __init__(self, indexes_list, batch_size: int = 10):
        self.batch_size = batch_size
        self.batches = [indexes_list[batch_size * i: batch_size * (i + 1)] for i in
                        range(math.ceil(len(indexes_list) / batch_size))]

    def __iter__(self):
        for batch in self.batches:
            print("batch:", batch)
            yield batch

    def __len__(self):
        return len(self.batches)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


def compute_recalls(eval_ds: TestDataset, queries_descriptors: np.ndarray, database_descriptors: np.ndarray,
                    output_folder: str = None, num_preds_to_save: int = 0,
                    save_only_wrong_preds: bool = True, logger: NeptuneLogger = None) -> Tuple[np.ndarray, str]:
    """Compute the recalls given the queries and database descriptors. The dataset is needed to know the ground truth
    positives for each query."""

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(queries_descriptors.shape[1])
    faiss_index.add(database_descriptors)
    del database_descriptors

    if logger is None:
        logging.debug("Calculating recalls")

    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    # For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])

    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, output_folder,
                                  save_only_wrong_preds, logger=logger)

    return recalls, recalls_str
