import faiss
import logging
from typing import Tuple
from pytorch_lightning.loggers import NeptuneLogger
import visualizations
from datasets.test_dataset import TestDataset
import numpy as np

# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


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
