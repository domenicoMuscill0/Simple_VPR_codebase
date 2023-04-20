from collections import defaultdict
import faiss
import logging
from typing import Tuple
import pytorch_lightning as pl
from pytorch_metric_learning import losses
from matplotlib import pyplot as plt
from torch.utils.data import BatchSampler
import visualizations
from datasets.test_dataset import TestDataset
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

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
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        # What if we mix the feature maps by using skip connections?
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class ProxyHead(nn.Module):
    def __init__(self, out_dim: int = 128, in_dim: int = 512,
                 n_classes: int = 23, n_layers: int = 3):
        super().__init__()
        a, b = in_dim, -np.log(out_dim / in_dim) / n_layers
        dims = np.ceil(a * np.exp(-b * np.arange(n_layers + 1))).astype(np.int16)
        self.model = [nn.Flatten()] + \
                     [nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU()) for i in range(n_layers)] + \
                     [nn.BatchNorm1d(out_dim)]
        self.model = nn.Sequential(*self.model)
        self.loss_fn = losses.ArcFaceLoss(n_classes, out_dim)
        self.optimizer = torch.optim.Adam(self.loss_fn.parameters(), lr=0.001, weight_decay=0.001)

    def forward(self, images):
        descriptors = self.model(images)
        return descriptors

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, compressed_descriptors, labels):
        loss = self.loss_fn(compressed_descriptors, labels)
        return loss

    def fit(self, descriptors, labels, epochs: int = 20):
        for i in range(epochs):
            compressed_descriptors = self(descriptors)
            loss = self.loss_fn(compressed_descriptors, labels)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()


class ProxyBank:
    """This class stores the places' proxies together with their identifier
       and performs exhaustive search on the index to retrieve the mini-batch sampling pool."""

    def __init__(self, img_per_place, bank=None, dim: int = 128, random_state: int = None):
        if bank is None:
            bank = defaultdict(ProxyBank.Proxy)
        self.__bank = bank
        np.random.seed(random_state)
        self.__index = ProxyBank.Index(faiss.IndexIDMap(faiss.IndexFlatIP(dim)))
        self.img_per_place = img_per_place
        self.n_batches = 23

    def set_n_batches(self, n_batches):
        self.n_batches = n_batches

    def update_bank(self, descriptors, labels):
        """This method adds descriptors and labels retrieved at each batch training step end to the underlying bank
           and the proxy index is updated."""
        # print("Le labels nella banca:", labels)
        for d, l in zip(descriptors, labels):
            self.__bank[l.item()] = self.__bank[l.item()] + ProxyBank.Proxy(d)
        self.__index.reset()
        bank = tuple(map(lambda it: it.get().cpu().detach(), self.__bank.values()))
        ids = list(self.__bank.keys())
        # print("Ecchime", ids)
        self.__index.add_with_ids(torch.stack(bank), ids)

    def sample_places(self, return_descriptors=False):
        """This function returns the pair (L, c_j) sampled from the underlying index, where:
           c_j: is the identifier of the sampled place for the next mini-batch
           L: is a list of np.array/torch.Tensor [l_1,...,l_M]
           M: number of places per mini-batch
           l_i: identifier of place P_i that is most similar to the extracted place proxy descriptor c_j"""
        n = len(self.__bank)
        if n == 0:
          n_places = 23  # counted from gsv_xs
          return np.random.randint(low=0, high=n_places, size=(self.n_batches, self.img_per_place)), None
        # print("Valori:", n, list(map(lambda v: v.get(), self.__bank.values())))
        c_k = np.random.choice(list(map(lambda v: v.get().cpu().detach(), self.__bank.values())), self.n_batches).tolist()
        # print("Le labels nella banca sono:",list(map(lambda k: k, self.__bank.keys())))
        return self.__index[c_k] if not return_descriptors else self.__index[c_k], c_k

    def __len__(self):
        return len(self.__bank.values())

    class Proxy:
        def __init__(self, tensor: torch.Tensor = None, n: int = 1, dim: int = 128):
            if tensor is None:
                self.__arch = torch.zeros(dim)
            else:
                self.__arch = tensor
            self.__arch = self.__arch.cuda()
            self.__n = n

        @property
        def get_shape(self):
            return self.__arch.shape

        def get(self):
            return self.__arch / self.__n

        def __add__(self, other):
            return ProxyBank.Proxy(tensor=self.__arch + other.__arch, n=self.__n + other.__n)

    class Index:
        def __init__(self, index: faiss.Index, top_places: int = 8,  # top_places must be batch_size / img_per_place
                     similarity="faiss-kNN", **kwargs):
            self._index = index  # Check if it is better L2 or Inner Product
            self.top_places = top_places

            if not isinstance(similarity, str):
                self.__custom_sim = True
                self.similarity = similarity
            else:
                self.__custom_sim = False

        def __getitem__(self, items):
            # We implement without deleting already selected places
            # Returns the ids and not the descriptors
            items = torch.stack(items)
            _, predictions = self._index.search(items, self.top_places)  # returns top top_places for each query
            # print("AOOOOOOOOOOOOOOOOOO1", len(items), predictions)
            if not self.__custom_sim:
                return predictions

        def reset(self):
            self._index.reset()

        def add_with_ids(self, bank, ids):
            self._index.add_with_ids(bank, ids)


class ProxyBatchSampler(BatchSampler):
    def __init__(self, sampler=None, data_labels=torch.arange(20), bank=ProxyBank(img_per_place=8),
                 batch_size: int = 64, img_per_place: int = 8, drop_last=True):
        super().__init__(sampler, batch_size, drop_last)
        assert batch_size % img_per_place == 0
        self.data_labels = data_labels
        self.bank = bank
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.drop_last = drop_last

    def set_labels(self, labels, shuffle=True):
        if shuffle:
            np.random.shuffle(labels)
        self.data_labels = labels

    def __iter__(self):
        # choices contains an entire epoch in the form of sequences of place ids. The dataset will extract img_per_place
        # images for each place
        choices, _ = self.bank.sample_places(return_descriptors=False)
        # print("AOOOOOOOOOOOOOOOOOO2", choices)
        for topM_places_ids in np.random.permutation(choices):
            # batch = [np.random.permutation(np.where(self.data_labels == id))[:self.num_choices] for id in
            #          topM_places_ids]
            # print("AOOOOOOOOOOOOOOO3", topM_places_ids)
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


def compute_recalls(eval_ds: TestDataset, queries_descriptors: np.ndarray, database_descriptors: np.ndarray,
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
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, output_folder, save_only_wrong_preds)

    return recalls, recalls_str
