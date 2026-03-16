'''
CUDA_VISIBLE_DEVICES=3 python src/moment_classification.py --dataset ctu
CUDA_VISIBLE_DEVICES=3 python src/moment_classification.py --dataset emg
CUDA_VISIBLE_DEVICES=3 python src/moment_classification.py --dataset had
CUDA_VISIBLE_DEVICES=3 python src/moment_classification.py --dataset har
CUDA_VISIBLE_DEVICES=3 python src/moment_classification.py --dataset rwc
CUDA_VISIBLE_DEVICES=3 python src/moment_classification.py --dataset tee
'''

import argparse
import os
from pprint import pprint
import math
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from momentfm import MOMENTPipeline
from sklearn.preprocessing import StandardScaler
from momentfm.utils.data import load_from_tsfile

from utils.file_io import save_embeddings


# -------------------------
# Config
# -------------------------
CONFIG = {
    "ctu": {"num_class": 2, "n_channels": 1, "series_len": 720,  "n_train": 250,   "n_test": 250},
    "emg": {"num_class": 3, "n_channels": 1, "series_len": 1500, "n_train": 63,    "n_test": 15},
    "had": {"num_class": 12, "n_channels": 6, "series_len": 128, "n_train": 34779, "n_test": 7929},
    "har": {"num_class": 6, "n_channels": 3, "series_len": 206,  "n_train": 4680,  "n_test": 2520},
    "rwc": {"num_class": 2, "n_channels": 1, "series_len": 4000, "n_train": 25499, "n_test": 4501},
    "tee": {"num_class": 7, "n_channels": 1, "series_len": 319,  "n_train": 35,    "n_test": 42},
}

W = 512
PATCH_LEN = 8


# -------------------------
# Data helpers
# -------------------------
def _to_NCT(x: np.ndarray) -> np.ndarray:
    """
    Convert possible shapes to (N, C, T).
    Common possibilities from ts loaders: (N, 1, T) or (N, T, C) or (N, C, T).
    """
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got {x.shape}")

    N, A, B = x.shape

    # Heuristic: channels usually small (<= 16). Time usually bigger.
    if A <= 16 and B > A:
        # (N, C, T)
        return x
    if B <= 16 and A > B:
        # (N, T, C) -> (N, C, T)
        return np.transpose(x, (0, 2, 1))

    # If ambiguous, default to assuming (N, C, T)
    return x


class TSFileDataset(Dataset):
    """
    Loads a dataset from .ts files and returns (C,T) + label.
    Scaler is fit on train split and passed into test split.
    """
    def __init__(self, dataset_name: str, split: str, scaler: StandardScaler = None):
        assert split in {"train", "test"}
        self.dataset_name = dataset_name
        self.split = split

        base = f"/raid/hdd249/data/moment/{dataset_name.lower()}/{dataset_name.upper()}"
        path = f"{base}_{split.upper()}.ts"

        X, y = load_from_tsfile(path)
        X = _to_NCT(X)  # (N,C,T)
        y = y.astype(int)

        self.X = X
        self.y_raw = y
        self.y = None  # filled in main() using train label map
        self.N, self.C, self.T = X.shape

        # scaling (fit on train, reuse on test)
        if scaler is None:
            scaler = StandardScaler()
            # Fit per-channel: reshape to (N*T, C)
            scaler.fit(np.transpose(X, (0, 2, 1)).reshape(-1, self.C))
        self.scaler = scaler

        X_scaled = self.scaler.transform(np.transpose(X, (0, 2, 1)).reshape(-1, self.C))
        X_scaled = X_scaled.reshape(self.N, self.T, self.C)
        X_scaled = np.transpose(X_scaled, (0, 2, 1))  # back to (N,C,T)
        self.X = X_scaled.astype(np.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.y is None:
            raise RuntimeError("TSFileDataset.y is None. Set it in main() using train label map.")
        return self.X[idx], int(self.y[idx])


def make_label_map(train_labels: np.ndarray):
    labels = np.unique(train_labels)
    return {int(l): i for i, l in enumerate(labels)}


def apply_label_map(labels: np.ndarray, label_map: dict):
    return np.vectorize(lambda z: label_map[int(z)])(labels).astype(int)


class WindowedLen512Dataset(Dataset):
    """
    Turns a (C,T) series into window tensors:
      windows: (K, C, 512)
      masks:   (K, 512)
      lengths: (K,)
      y:       int
    """
    def __init__(self, base_ds, W=512):
        self.base = base_ds
        self.W = W

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]  # x: (C,T)
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]     # (1,T)
        if x.ndim != 2:
            raise ValueError(f"Expected (C,T) or (T,), got {x.shape}")

        C, L = x.shape
        W = self.W

        windows, masks, lengths = [], [], []
        start = 0
        while start < L:
            end = min(start + W, L)
            chunk = x[:, start:end]     # (C, len_i)
            len_i = chunk.shape[1]

            pad = W - len_i
            if pad > 0:
                chunk_padded = np.pad(chunk, ((0, 0), (pad, 0)), mode="constant")  # left-pad time
            else:
                chunk_padded = chunk

            mask = np.ones((W,), dtype=np.float32)
            if pad > 0:
                mask[:pad] = 0.0

            windows.append(chunk_padded.astype(np.float32))  # (C,W)
            masks.append(mask)
            lengths.append(float(len_i))

            start += W

        windows = torch.from_numpy(np.stack(windows, axis=0))   # (K,C,W)
        masks = torch.from_numpy(np.stack(masks, axis=0))       # (K,W)
        lengths = torch.tensor(lengths, dtype=torch.float32)    # (K,)
        return windows, masks, lengths, int(y)


# -------------------------
# Collate: batch at window-level
# -------------------------
def collate_windowed(batch):
    """
    batch: list of (windows, masks, lengths, y)
      windows: (K_i, C, 512)
      masks:   (K_i, 512)
      lengths: (K_i,)
      y:       int
    Returns a dict with flattened windows and a mapping back to series ids.
    """
    all_windows = []
    all_masks = []
    all_lengths = []
    series_ids = []
    labels = []

    for series_idx, (windows, masks, lengths, y) in enumerate(batch):
        K = windows.shape[0]
        all_windows.append(windows)
        all_masks.append(masks)
        all_lengths.append(lengths)
        series_ids.append(torch.full((K,), series_idx, dtype=torch.long))
        labels.append(y)

    return {
        "windows": torch.cat(all_windows, dim=0),     # (sumK, C, 512)
        "masks": torch.cat(all_masks, dim=0),         # (sumK, 512)
        "lengths": torch.cat(all_lengths, dim=0),     # (sumK,)
        "series_ids": torch.cat(series_ids, dim=0),   # (sumK,)
        "labels": torch.tensor(labels, dtype=torch.long),  # (B,)
    }


# -------------------------
# Embedding extraction (batched)
# -------------------------
def _pool_patches_to_window(e: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    e: (Nw, Npatch, D) or (Nw, D)
    lengths: (Nw,)
    Returns: (Nw, D)
    Uses left-pad aware patch pooling: keep last ceil(len_i/PATCH_LEN) patches.
    """
    if e.ndim == 2:
        return e
    if e.ndim != 3:
        raise ValueError(f"Unexpected embeddings shape: {tuple(e.shape)}")

    pooled = []
    for i in range(e.shape[0]):
        len_i = int(lengths[i].item())
        n_real = max(1, math.ceil(len_i / PATCH_LEN))
        pooled.append(e[i, -n_real:, :].mean(dim=0))
    return torch.stack(pooled, dim=0)  # (Nw, D)


def get_embedding_window_avg_batched(model, dataloader):
    """
    Returns:
      embeddings: (N, D)
      labels:     (N,)
    """
    final_embs = []
    final_labels = []

    device = torch.device("cuda")

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            windows = batch["windows"].to(device).float()     # (Nw, C, 512)
            masks = batch["masks"].to(device).float()         # (Nw, 512)
            lengths = batch["lengths"].to(device).float()     # (Nw,)
            series_ids = batch["series_ids"].to(device)       # (Nw,)
            labels = batch["labels"].cpu().numpy()            # (B,)

            out = model(x_enc=windows, input_mask=masks)
            e = out.embeddings                                # (Nw, Npatch, D) or (Nw, D)

            # Patch -> window embedding: (Nw, D)
            e = _pool_patches_to_window(e, lengths)

            # Window -> series embedding (length-weighted)
            B = labels.shape[0]
            D = e.shape[1]
            series_embs = torch.zeros((B, D), device=device)
            series_wsum = torch.zeros((B,), device=device)

            # Simple scatter-add loop (fast enough; optimize later if needed)
            for i in range(e.shape[0]):
                sid = int(series_ids[i].item())
                w = lengths[i]
                series_embs[sid] += w * e[i]
                series_wsum[sid] += w

            series_embs = series_embs / series_wsum[:, None]

            final_embs.append(series_embs.detach().cpu().numpy())
            final_labels.append(labels)

    return np.concatenate(final_embs, axis=0), np.concatenate(final_labels, axis=0)


# -------------------------
# Args / main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MOMENT Embedding Extraction (windowed + batched)")
    p.add_argument("--dataset", choices=list(CONFIG.keys()), type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64, help="series batch size (windows are flattened)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Configuration | {args.dataset}")
    pprint(CONFIG[args.dataset])
    print(f"Loading MOMENT-1 model for {args.dataset} dataset...")

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            "task_name": "classification",
            "n_channels": CONFIG[args.dataset]["n_channels"],
            "num_class": CONFIG[args.dataset]["num_class"],
        },
    )
    model.init()
    model.to("cuda").float()
    model.eval()

    # ----- load + normalize + label-map (train -> test)
    train_base = TSFileDataset(args.dataset, "train", scaler=None)
    label_map = make_label_map(train_base.y_raw)
    train_base.y = apply_label_map(train_base.y_raw, label_map)

    test_base = TSFileDataset(args.dataset, "test", scaler=train_base.scaler)
    test_base.y = apply_label_map(test_base.y_raw, label_map)

    # ----- windowing wrapper
    train_dataset = WindowedLen512Dataset(train_base, W=W)
    test_dataset = WindowedLen512Dataset(test_base, W=W)

    # quick sanity
    w0, m0, lens0, y0 = train_dataset[0]
    print("sample0:", "windows", tuple(w0.shape), "mask", tuple(m0.shape), "lengths", lens0.tolist(), "label", y0)

    # ----- dataloaders (NOTE: collate_fn is the whole point)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_windowed,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_windowed,
        num_workers=0,
        pin_memory=True,
    )

    # ----- embeddings
    train_embeddings, train_labels = get_embedding_window_avg_batched(model, train_loader)
    test_embeddings, test_labels = get_embedding_window_avg_batched(model, test_loader)

    print("train:", train_embeddings.shape, train_labels.shape)
    print("test :", test_embeddings.shape, test_labels.shape)

    out_dir = f"/raid/hdd249/data/sample_features/moment/{args.dataset}"
    print(f"Saving embeddings to {out_dir} ...")
    os.makedirs(out_dir, exist_ok=True)

    save_embeddings(
        save_path=out_dir,
        train_embed=train_embeddings,
        test_embed=test_embeddings,
        overwrite=True,
    ) 
    print(f"✅ Saved train/test embeddings to: {out_dir}")

if __name__ == "__main__":
    main()
