import os

import numpy as np

DEFAULT_PRETRAINED_EXPID = {
    ("ECG", False): 10,
    ("ECG", True): 15,
    ("METR-LA", False): 20,
    ("METR-LA", True): 25,
    ("SOLAR", False): 30,
    ("SOLAR", True): 35,
    ("TRAFFIC", False): 40,
    ("TRAFFIC", True): 45,
}

def infer_dataset_shape(dataset_dir: str):

    train_path = os.path.join(dataset_dir, "train.npz")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Cannot infer dataset shape because {train_path} does not exist."
        )

    with np.load(train_path) as train_data:
        x = train_data["x"]

    if x.ndim != 4:
        raise ValueError(
            f"Expected train.npz['x'] to be 4-D, but got shape {x.shape}."
        )

    num_nodes = int(x.shape[2])
    in_dim = int(x.shape[3])
    return num_nodes, in_dim

def resolve_pretrained_expid(dataset: str, mask_layer: bool, pretrained_expid=None):
    if pretrained_expid is not None:
        return int(pretrained_expid)

    key = (dataset, bool(mask_layer))
    if key in DEFAULT_PRETRAINED_EXPID:
        return DEFAULT_PRETRAINED_EXPID[key]

    raise ValueError(
        "No default pretrained expid is defined for "
        f"dataset={dataset}, mask_layer={mask_layer}. "
        "Please provide --pretrained_expid explicitly."
    )
