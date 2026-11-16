import random

import numpy as np
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from .config import (
    DATASET_ROOT,
    CPU_NUM_THREADS,
)


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def configure_cpu():

    torch.set_num_threads(
        CPU_NUM_THREADS
    )


def load_dataset(name):

    dataset = Planetoid(
        root=f"{DATASET_ROOT}/{name}",
        name=name,
        transform=NormalizeFeatures(),
    )

    data = dataset[0]

    return dataset, data


def dataset_info(dataset, data):

    return {
        "dataset": dataset.name,

        "nodes": data.num_nodes,

        "edges": data.edge_index.size(1),

        "features": dataset.num_features,

        "classes": dataset.num_classes,

        "train_nodes":
            int(data.train_mask.sum()),

        "val_nodes":
            int(data.val_mask.sum()),

        "test_nodes":
            int(data.test_mask.sum()),
    }