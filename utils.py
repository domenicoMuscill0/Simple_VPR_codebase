
from collections import defaultdict
import faiss
import logging
import numpy as np
from typing import Tuple
import pytorch_lightning as pl
import torch
from pytorch_metric_learning import losses
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, BatchSampler
import visualizations

# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


class ProxyHead(nn.Module):
    def __init__(self, out_dim: int = 128, in_dim: int = 512, n_layers: int = 3):
        super().__init__()
        a, b = in_dim, -np.log(out_dim / in_dim) / n_layers
        dims = np.ceil(a * np.exp(-b * np.arange(n_layers + 1))).astype(np.int16)
        self.model = [nn.Flatten()] + \
                     [nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU()) for i in range(n_layers)] + \
                     [nn.BatchNorm1d(out_dim)]
        self.model = nn.Sequential(*self.model)
        self.loss_fn = losses.VICRegLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)

    def forward(self, images):
        descriptors = self.model(images)
        return descriptors

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        loss = self.loss_fn(descriptors, labels)
        return loss

    def fit(self, descriptors, labels, epochs: int = 20):
        for i in range(epochs):
            compressed_descriptors = self(descriptors)
            loss = self.loss_fn(compressed_descriptors, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class ProxyBank:
    """This class stores the places' proxies together with their identifier
       and performs exhaustive search on the index to retrieve the mini-batch sampling pool."""

    def __init__(self, M, bank=None, dim: int = 128, random_state: int = None):
        if bank is None:
            bank = defaultdict(ProxyBank.Proxy)
        self.__bank = bank
        np.random.seed(random_state)
        self.__index = ProxyBank.Index(faiss.IndexIDMap(faiss.IndexFlatIP(dim)))
        self.M = M

    def update_bank(self, descriptors, labels):
        """This method adds descriptors and labels retrieved at each batch training step end to the underlying bank
           and the proxy index is updated."""
        for d, l in zip(descriptors, labels):
            self.__bank[l] = self.__bank[l] + ProxyBank.Proxy(d)
        self.__index.reset()
        self.__index.add_with_ids(self.__bank.values(), self.__bank.keys())

    def sample_places(self, return_descriptors=False):
        """This function returns the pair (L, c_j) sampled from the underlying index, where:
           c_j: is the identifier of the sampled place for the next mini-batch
           L: is a list of np.array/torch.Tensor [l_1,...,l_M]
           M: number of places per mini-batch
           l_i: identifier of place P_i that is most similar to the extracted place proxy descriptor c_j"""
        c_k = np.random.permutation(list(self.__bank.values()))
        return self.__index[c_k] if not return_descriptors else self.__index[c_k], c_k

    def __len__(self):
        return len(self.__bank.values())

    class Proxy:
        def __init__(self, tensor: torch.Tensor = None, n: int = 0, dim: int = 128):
            if tensor is None:
                self.__arch = torch.zeros(dim)
            else:
                self.__arch = tensor
            self.__n = n

        def get(self, numpy: bool = False):
            return self.__arch / self.__n if not numpy else (self.__arch / self.__n).cpu().numpy().astype(np.float32)

        def __add__(self, other):
            return ProxyBank.Proxy(tensor=self.__arch + other.__arch, n=self.__n + other.__n)

    class Index:
        def __init__(self, index: faiss.Index, M: int = 10,
                     similarity="faiss-kNN", **kwargs):
            self._index = index  # Check if it is better L2 or Inner Product
            self.M = M
            self.__n = 0

            if not isinstance(similarity, str):
                self.__custom_sim = True
                self.similarity = similarity
            else:
                pass

        def __getitem__(self, items):
            # We implement without deleting already selected places
            # Returns the ids and not the descriptors
            n = len(items)
            if n == 0:
                n_places = 23  # counted from gsv_xs
                return np.random.randint(low=0, high=n_places, size=(n_places // 2, self.M))
            items = torch.stack(list(map(lambda it: it.get(), items)))
            _, predictions = self._index.search(items, self.M)
            if not self.__custom_sim:
                return predictions[0] if n == 1 else predictions

        def reset(self):
            self._index.reset()

        def add_with_ids(self, bank, ids):
            n = len(bank)
            self.__n += n
            self._index.add_with_ids(n, bank, ids)


class ProxyBatchSampler(BatchSampler):
    def __init__(self, data_labels=torch.arange(20), bank=ProxyBank(M=8), batch_size: int = 64,
                 M: int = 8, drop_last=True):
        assert batch_size % M == 0
        self.data_labels = data_labels
        self.bank = bank
        self.batch_size = batch_size
        self.num_choices = batch_size // M
        self.M = M
        self.drop_last = drop_last

    def set_labels(self, labels, shuffle=True):
        if shuffle:
            np.random.shuffle(labels)
        self.data_labels = labels

    def __iter__(self):
        choices = self.bank.sample_places(return_descriptors=False)
        for topM_places_ids in np.random.permutation(choices).astype(object):
            # batch = [np.random.permutation(np.where(self.data_labels == id))[:self.num_choices] for id in
            #          topM_places_ids]
            yield topM_places_ids

    def __len__(self):
        return self.batch_size


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


class PrintDescriptor(pl.Callback):
    def __init__(self, n_images: int = 2, im_path: str = "."):
        self.n_images = n_images
        self.im_path = im_path

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,  # Training step output
            batch,
            batch_idx: int,
            dataloader_idx: int = 0):
        images, labels = batch
        for i in range(self.n_images):
            descriptor = pl_module(images[i])
            descriptor = descriptor.reshape(descriptor.size(-2), descriptor.size(-1), 3)
            plt.imsave(self.im_path + f"/descriptors/P:{labels[i]}_B:{batch_idx}", descriptor)
            # Add an epoch variable to see how descriptors change?
            plt.imshow(descriptor)


def compute_recalls(eval_ds: Dataset, queries_descriptors: np.ndarray, database_descriptors: np.ndarray,
                    output_folder: str = None, num_preds_to_save: int = 0,
                    save_only_wrong_preds: bool = True) -> Tuple[np.ndarray, str]:
    """Compute the recalls given the queries and database descriptors. The dataset is needed to know the ground truth
    positives for each query."""

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(queries_descriptors.shape[1])
    faiss_index.add(database_descriptors)
    del database_descriptors

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    #### For each query, check if the predictions are correct
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
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, output_folder, save_only_wrong_preds)

    return recalls, recalls_str
