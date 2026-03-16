'''
python ./src/data_management/squish_had.py
'''

import os
import json
import sys
import numpy as np
from typing import Dict, List, Tuple

sys.path.append("./src")
from utils.loaders import load_train_test, Split

# --- CONFIGURATION & MAPPINGS ---
HAD_TO_HAR_MAP = {
    1: 0, 2: 0, 3: 0,  # All Walking -> WALKING
    4: 1,              # Walking Upstairs -> WALKING_UPSTAIRS
    5: 2,              # Walking Downstairs -> WALKING_DOWNSTAIRS
    8: 3,              # Sitting -> SITTING
    9: 4,              # Standing -> STANDING
    10: 5,             # Sleeping -> LAYING
}

HAR_LABEL_NAMES = {
    0: "WALKING", 1: "WALKING_UPSTAIRS", 2: "WALKING_DOWNSTAIRS",
    3: "SITTING", 4: "STANDING", 5: "LAYING"
}

def ensure_directory_exists(path: str):
    os.makedirs(path, exist_ok=True)

def _generate_few_shots(y: np.ndarray, k: int = 5) -> Dict[int, List[int]]:
    """Generates 5 indices per class for the class_shots.json."""
    out = {}
    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        # In case a class has fewer than k samples
        sel = np.random.choice(idxs, size=min(k, len(idxs)), replace=False)
        out[int(label)] = sel.tolist()
    return out

def save_split_from_object(output_dir: str, train: Split, test: Split):
    """Saves Split objects into the directory format expected by the loader."""
    ensure_directory_exists(output_dir)

    # 1. Save data arrays
    np.savez(os.path.join(output_dir, "train.npz"), X_train=train.X, y_train=train.y)
    np.savez(os.path.join(output_dir, "test.npz"), X_test=test.X, y_test=test.y)

    # 2. Build and save label_maps.json
    # Logic: Letter 'A'=0, 'B'=1, etc.
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sorted_ids = sorted(list(HAR_LABEL_NAMES.keys()))
    
    label_maps = {
        "letter_to_id": {letters[i]: str(cid) for i, cid in enumerate(sorted_ids)},
        "id_to_letter": {str(cid): letters[i] for i, cid in enumerate(sorted_ids)},
        "id_to_name": {str(cid): name for cid, name in HAR_LABEL_NAMES.items()}
    }
    with open(os.path.join(output_dir, "label_maps.json"), "w") as f:
        json.dump(label_maps, f, indent=2)

    # 3. Save few-shot indices
    shots = _generate_few_shots(train.y, k=5)
    with open(os.path.join(output_dir, "class_shots.json"), "w") as f:
        json.dump({str(k): v for k, v in shots.items()}, f, indent=2)

    # 4. Save metadata
    with open(os.path.join(output_dir, "general_question.txt"), "w") as f:
        f.write(train.general_question or "Determine the human activity based on accelerometer data.")

    print(f"✔ Dataset saved to: {output_dir}")

def squish_had_to_har(train: Split, test: Split) -> tuple[Split, Split]:
    results = []
    for split in [train, test]:
        # Filter for only classes we care about
        mask = np.isin(split.y, list(HAD_TO_HAR_MAP.keys()))
        
        # Slicing X to keep only first 3 channels (Acc X, Y, Z)
        new_X = split.X[mask][:, :, :3]
        
        # Remap y labels
        new_y = np.array([HAD_TO_HAR_MAP[val] for val in split.y[mask]], dtype=np.int64)
        
        results.append(Split(
            X=new_X, y=new_y, idx=np.arange(len(new_y)),
            dataset="har_compat",
            general_question="Determine the human activity based on tri-axial accelerometer data."
        ))
    return results[0], results[1]

def main():
    # 1. Load raw datasets
    print("Loading datasets...")
    HADtrain_raw, HADtest_raw = load_train_test(input_folder="/raid/hdd249/data/samples/had", n_shots=0)
    HARtrain_raw, HARtest_raw = load_train_test(input_folder="/raid/hdd249/data/samples/har", n_shots=0)

    # 2. Process HAD -> HAR-compatible format (Squish labels + Accel only)
    print("Squishing HAD...")
    HADtrain_sq, HADtest_sq = squish_had_to_har(HADtrain_raw, HADtest_raw)

    # 3. Standardize HAR (Accel only)
    print("Standardizing HAR features...")
    HARtrain_raw.X = HARtrain_raw.X[:, :, :3]
    HARtest_raw.X = HARtest_raw.X[:, :, :3]

    # --- CROSS-DATASET SAVING ---

    # Scenario A: Train on HAD, Test on HAR (trHADteHAR)
    # This tests if HAD training generalizes to HAR data
    print("\nSaving Scenario: Train HAD -> Test HAR")
    save_split_from_object(
        output_dir="/raid/hdd249/data/samples/trHADteHAR", 
        train=HADtrain_sq, 
        test=HARtest_raw
    )

    # Scenario B: Train on HAR, Test on HAD (trHARteHAD)
    # This tests if HAR training generalizes to HAD data
    print("Saving Scenario: Train HAR -> Test HAD")
    save_split_from_object(
        output_dir="/raid/hdd249/data/samples/trHARteHAD", 
        train=HARtrain_raw, 
        test=HADtest_sq
    )

    print("\n--- Generalizability Splits Prepared ---")
    print(f"trHADteHAR: Train samples={len(HADtrain_sq)}, Test samples={len(HARtest_raw)}")
    print(f"trHARteHAD: Train samples={len(HARtrain_raw)}, Test samples={len(HADtest_sq)}")

if __name__ == "__main__":
    main()