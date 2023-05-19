import math
from collections import defaultdict

import faiss
import numpy as np
import torch
from pytorch_metric_learning import losses
from torch import nn
from torch.utils.data import Sampler


class ProxyHead(nn.Module):
    def __init__(self, out_dim: int = 32, in_dim: int = 128,
                 n_classes: int = 23, n_layers: int = 3):
        super().__init__()
        a, b = in_dim, -np.log(out_dim / in_dim) / n_layers
        dims = np.ceil(a * np.exp(-b * np.arange(n_layers + 1))).astype(np.int16)
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
    return lambda: ProxyBank.Proxy(dim=dim)


class ProxyBank:
    """This class stores the places' proxies together with their identifier
       and performs exhaustive search on the index to retrieve the mini-batch sampling pool."""

    def __init__(self, k: int = 10, dim: int = 128):
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

        def get(self) -> torch.Tensor:
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
