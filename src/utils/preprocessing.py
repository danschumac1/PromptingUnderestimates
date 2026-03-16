
import json
import re
import string

import numpy as np

from utils.constants import LABEL_MAPPING, LEGEND_MAPPINGS, TASK_DESCRIPTION

def extract_letter_to_idx(model_output: str, mapping: dict[str, int]) -> tuple[str, int]:
    """
    Extract the predicted label (e.g., 'A', 'B', 'C', ...) from a model output string
    and map it to an integer id using `mapping`.

    Heuristics:
      1. Prefer explicit patterns like "The answer is [X]" or "Answer: [X]".
      2. If none found, fall back to the *last* bracketed token [X] in the text
         that appears in `mapping`.
      3. If nothing valid is found, return ("no valid letter found", -1).
    """
    # 1. Look for explicit "answer" patterns
    answer_patterns = [
        r"[Tt]he answer is\s*\[([^\[\]]+)\]",
        r"[Ff]inal answer\s*[:\-]?\s*\[([^\[\]]+)\]",
        r"[Aa]nswer\s*[:\-]?\s*\[([^\[\]]+)\]",
    ]
    for pat in answer_patterns:
        m = re.search(pat, model_output)
        if m:
            cand = m.group(1).strip()
            if cand in mapping:
                return cand, mapping[cand]

    # 2. Fallback: scan all bracketed tokens and use the LAST one that maps
    candidates = re.findall(r"\[([^\[\]]+)\]", model_output)
    for cand in reversed(candidates):
        cand = cand.strip()
        if cand in mapping:
            return cand, mapping[cand]

    # 3. Nothing matched
    return "no valid letter found", -1



def serialize_ts(
    X: np.ndarray,
    max_chars: int = 24000,
    decimals: int = 3,
    return_as_list: bool = False,
) -> str | list:
    """
    Serialize a time series to JSON, downsampling along the *longest* dimension
    until the string is under `max_chars` characters.

    If return_as_list=True, returns a Python list (possibly downsampled).
    Otherwise returns a JSON string.
    """

    X_round = np.round(X, decimals=decimals)

    # First serialization attempt: no downsampling
    as_list = X_round.tolist()
    s = json.dumps(as_list)
    orig_len = len(s)

    # ✅ FIX: respect return_as_list even when it fits
    if orig_len <= max_chars:
        return as_list if return_as_list else s

    # -------------------------------------------------------------
    # Too large → downsample along the *longest* axis (likely time)
    # -------------------------------------------------------------
    if X_round.ndim == 1:
        time_axis = 0
    else:
        time_axis = int(np.argmax(X_round.shape))

    T = X_round.shape[time_axis]
    est_factor = int(np.ceil(orig_len / max_chars))
    factor = max(est_factor, 1)

    def downsample(X_arr: np.ndarray, step: int) -> np.ndarray:
        slicers = [slice(None)] * X_arr.ndim
        slicers[time_axis] = slice(0, X_arr.shape[time_axis], step)
        return X_arr[tuple(slicers)]

    X_ds = downsample(X_round, factor)
    s_ds = json.dumps(X_ds.tolist())

    while len(s_ds) > max_chars and X_ds.shape[time_axis] > 2:
        factor += 1
        X_ds = downsample(X_round, factor)
        s_ds = json.dumps(X_ds.tolist())

    final_len = len(s_ds)
    final_T = X_ds.shape[time_axis]

    # ✅ FIX: warnings.warn (not Warning.warn)
    warnings.warn(
        f"[serialize_ts] Time series too long → downsampled.\n"
        f"  Original: T={T}, chars={orig_len}\n"
        f"  Final:    T={final_T}, chars={final_len}\n"
        f"  Axis:     time_axis={time_axis}\n"
        f"  Factor:   every {factor}-th timestep kept\n"
        f"  Threshold: max_chars={max_chars}\n",
        category=UserWarning,
        stacklevel=2,
    )

    return X_ds.tolist() if return_as_list else s_ds


def letcs_transform(ts_list: list, precision: int = 3) -> str:
    formatted_steps = []
    for x in ts_list:
        s = f"{float(x):.{precision}f}"
        s = s.replace(".", "").replace(",", "")
        formatted_steps.append(" ".join(list(s)))
    return " , ".join(formatted_steps)


def letcs_transform_multivar(ts_2d, precision: int = 3) -> str:
    """
    Convert a multivariate time series (any ndim >= 1) into a single
    LETCS-style string by flattening all dimensions.
    """
    arr = np.asarray(ts_2d, dtype=float)   # now ts_2d is truly numeric list
    flat = arr.flatten().tolist()
    return letcs_transform(flat, precision=precision)


def _letters(n: int) -> str:
    letters = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(string.ascii_uppercase[rem])
    return "".join(reversed(letters))


def _sort_key_for_label_id(k):
    try:
        return (0, int(k))
    except (TypeError, ValueError):
        return (1, str(k))


def build_question_text(subset_key: str) -> str:
    key = subset_key.strip().upper()
    if key not in TASK_DESCRIPTION or key not in LABEL_MAPPING:
        raise ValueError(f"Unknown subset key '{subset_key}'. Valid keys: {sorted(LABEL_MAPPING.keys())}")

    task = TASK_DESCRIPTION[key].strip()
    labels = LABEL_MAPPING[key]
    sorted_items = sorted(labels.items(), key=lambda kv: _sort_key_for_label_id(kv[0]))
    label_texts = [v for _, v in sorted_items]
    options_lines = [f"[{_letters(i+1)}] {opt}" for i, opt in enumerate(label_texts)]
    options_block = "Your options are:\n\t" + "\n\t".join(options_lines)
    return f"{task} {options_block}" if task else options_block



def get_dim_names(dataset: str) -> list[str]:
    """
    Use LEGEND_MAPPINGS if available; otherwise fall back to dim_0, dim_1, ...
    Slugify names so they're safe as JSON keys (no spaces/hyphens).
    """
    legends = LEGEND_MAPPINGS.get(dataset.upper())
    if legends is None:
        legends=["time_series"]
    elif len(legends) == 1 or legends is None:
        legends = ["time_series"]

    return legends