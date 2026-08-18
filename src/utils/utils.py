import numpy as np

def generate_mask_mat(mat: np.ndarray, p: float = 0.1, missing_type: str = "MCAR"):

    if p < 0.0 or p > 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    missing_type = str(missing_type).upper()

    if p == 0:
        return np.zeros_like(mat, dtype=bool)
    if p == 1:
        return np.ones_like(mat, dtype=bool)

    if missing_type == "MCAR":
        return np.random.rand(*mat.shape) < p

    mat_float = np.asarray(mat, dtype=float)

    if missing_type == "MAR":

        axes = tuple(range(mat_float.ndim - 1)) if mat_float.ndim > 1 else None
        mean = np.nanmean(mat_float, axis=axes, keepdims=True)
        std = np.nanstd(mat_float, axis=axes, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        z = (mat_float - mean) / std

        base_prob = 1.0 / (1.0 + np.exp(-z))
    elif missing_type == "MNAR":

        data_min = np.nanmin(mat_float)
        data_max = np.nanmax(mat_float)
        if data_max == data_min:
            base_prob = np.full_like(mat_float, p, dtype=float)
        else:
            norm = (mat_float - data_min) / (data_max - data_min + 1e-8)

            base_prob = norm ** 2
    else:
        raise ValueError(f"Unsupported missing_type: {missing_type}")

    current_mean = float(np.nanmean(base_prob))
    if current_mean > 0:
        scale = p / current_mean
        prob = np.clip(base_prob * scale, 0.0, 1.0)
    else:
        prob = np.full_like(base_prob, p, dtype=float)

    return np.random.rand(*mat.shape) < prob
