"""
python ./src/data_management/enrich_rwc.py
Writes placeholder enrichment artifacts into ./data/samples/rwc/
"""

#!/usr/bin/env python
import os, json
import numpy as np

import sys; sys.path.append("./src")
from utils.constants import LABEL_MAPPING
from utils.preprocessing import _letters, _sort_key_for_label_id, build_question_text
from utils.loaders import load_train_test


def build_letter_maps(dataset: str):
    key = dataset.strip().upper()
    id_to_name = LABEL_MAPPING[key]
    items = sorted(id_to_name.items(), key=lambda kv: _sort_key_for_label_id(kv[0]))
    id_to_letter = {int(cid): _letters(i + 1) for i, (cid, _) in enumerate(items)}
    letter_to_id = {letter: cid for cid, letter in id_to_letter.items()}
    return id_to_letter, letter_to_id, id_to_name


def main():
    dataset = "rwc"

    out_dir = f"./data/samples/{dataset}/"
    data_dir = out_dir  # load from same split we enrich
    os.makedirs(out_dir, exist_ok=True)

    # Load train/test to know ordering
    _, test = load_train_test(data_dir, 10)  
    X_te = np.asarray(test.X)

    # Build maps + question
    id_to_letter, letter_to_id, id_to_name = build_letter_maps(dataset)
    question = build_question_text(dataset).strip()

    # ALWAYS K=10
    K = 10
    N_te = len(X_te)
    top_same = np.full((N_te, K), -1, dtype=np.int32)

    # Write artifacts
    np.save(os.path.join(out_dir, "top10_similar.npy"), top_same)

    with open(os.path.join(out_dir, "general_question.txt"), "w") as f:
        f.write(question)

    maps = {
        "letter_to_id": letter_to_id,  # {"A": 0, "B": 1, ...}
        "id_to_letter": {str(k): v for k, v in id_to_letter.items()},
        "id_to_name":   {str(k): v for k, v in id_to_name.items()},
    }
    with open(os.path.join(out_dir, "label_maps.json"), "w") as f:
        json.dump(maps, f, indent=2)

    print(f"[OK] wrote artifacts → {out_dir}")


if __name__ == "__main__":
    main()
