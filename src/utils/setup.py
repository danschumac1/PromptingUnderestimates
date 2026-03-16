# IMPORTS
import argparse
import os
import random
from typing import Union

import numpy as np

from utils.random_prompter import RandomLlamaVisionPrompter, RandomMistralVisionPrompter, RandomQwenVisionPrompter
from utils.prompters import LlamaVisionPrompter, MistralVisionPrompter, QwenVisionPrompter
from utils.loaders import Split, load_train_test
from utils.loggers import MasterLogger
from utils.prompters import VisionPrompter

# CONSTANTS
PROMPTER_CLASSES = {
    "llama": LlamaVisionPrompter,
    "mistral": MistralVisionPrompter,
    "qwen": QwenVisionPrompter,
    "random_llama": RandomLlamaVisionPrompter,
    "random_mistral": RandomMistralVisionPrompter,
    "random_qwen": RandomQwenVisionPrompter,
}

# HELPERS
def _build_tag(
    script: str,
    n_shots: int,
    include_user_text: bool,
    include_ts: bool,
    include_vis: bool,
    include_LETSCLike: bool,
    CoT: bool,
) -> str:
    assert script in ["embedding", "prompting"]

    flags = [
        ("ts", include_ts),
        ("vis", include_vis),
        ("lets", include_LETSCLike),
        ("ust", include_user_text),
    ]

    # dash-joined embedding components
    emb_part = "-".join(name for name, exists in flags if exists)
    assert emb_part, "At least one of ts/vis/lets/ust must be True"

    tag = f"{emb_part}_{n_shots}-shot"

    if CoT:
        tag += "_CoT.jsonl"
    else:
        tag += "_Direct.jsonl"

    return tag




# IMPORTABLE FUNCTIONS
def standard_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated answers (vision LLaMA).")
    # REQUIRED arguments
    parser.add_argument(
        "--dataset",
        choices=["ctu", "emg", "had", "har", "rwc", "tee", "trHARteHAD"],
        type=str,
        required=True,
        help="Dataset name (must match /raid/hdd249/data/samples/{dataset})",
    )
    parser.add_argument(
        "--model",
        choices=[
            "llama",
            "mistral",
            "qwen", 
            "random_llama",
            "random_mistral",
            "random_qwen", 
            ],
        type=str,
        required=True,
        help="Model to use for prompting",
    )
    parser.add_argument(
        "--CoT",
        choices=[0,1],
        type=int,
        default=0,
        help="Whether or not to use chain of thought prompting. CANNOT be used with few shot prompting."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing samples (used when n_shots == 0)",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=5,
        help="Number of few-shot examples to use in the prompt (0 = zero-shot)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        choices=[0,1],
        default=1
    )
    return parser



def setup(
        script:str,
        dataset:str, 
        model: str,
        n_shots:int,
        include_ts:bool,
        include_vis:bool,
        include_LETSCLike:bool,
        include_user_text:bool=True,
        sample:bool=True,
        CoT: bool = False
        ) -> tuple[str, Split, Split, MasterLogger, VisionPrompter]:
    supported_model_str = ", ".join(list(PROMPTER_CLASSES.keys()))
    assert model in PROMPTER_CLASSES.keys(), f"Model {model} not supported yet. Supported models include: {supported_model_str} "
    assert script in ["embedding", "prompting", "zero_shot_sc"], f"User imputed script as: {script}. Must be 'embedding', 'prompting', or 'zero_shot_sc'\n but got {script}"
    assert include_ts or include_vis or include_LETSCLike, \
        "At least one of 'include_ts' or 'include_vis' or 'include_LETSCLike' must be true"
    assert not (include_ts and include_LETSCLike), \
        "Only one of 'include_ts' or 'include_LETSCLike' can be true"

    random.seed(42)
    np.random.seed(42)
    
    PrompterCls: VisionPrompter = PROMPTER_CLASSES[model]
    prompter = PrompterCls()

    # Load train/test; loader attaches label_maps & shot_idxs
    train, test = load_train_test(
        # input_folder=f"/raid/hdd249/data/samples/{dataset}",
        input_folder=f"/raid/hdd249/data/samples/{dataset}",
        n_shots=n_shots,
    )

    logs_dir = f"./logs/{script}/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{dataset}.log")
    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=True,
    )

    if script == "prompting":
        out_str = "sample_generations" if sample else "generations"
        # out_dir = f"/raid/hdd249/data/{out_str}/{model}/{dataset}/visual_prompting/" 
        out_dir = f"/raid/hdd249/data/{out_str}/{model}/{dataset}/visual_promptingABLATION/" 
        os.makedirs(out_dir, exist_ok=True)
        tag = _build_tag(script, n_shots, include_user_text, include_ts, include_vis, include_LETSCLike, CoT)
        out_file = os.path.join(out_dir, tag)
        logger.info(f"Clearing output file {out_file}")
        with open(out_file, "w") as _:
            pass

    elif script == "embedding":
        out_str = "sample_features" if sample else "features"
        out_dir = f"/raid/hdd249/data/{out_str}/{model}/{dataset}/"
        tag = _build_tag(script, n_shots, include_user_text, include_ts, include_vis, include_LETSCLike, CoT)
        out_file = os.path.join(out_dir, tag)
        out_dir =out_file.replace(".jsonl", "")
        os.makedirs(out_dir, exist_ok=True)
        out_file = out_dir
        logger.info(f"Embeddings will be saved at/over {out_file} train*test_embeddings.npz")

    elif script == "zero_shot_sc":
        out_dir = f"/raid/hdd249/data/generations/{model}/{dataset}/self_consistency/"
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{n_shots}shot_{model}_self_consistency.jsonl"
        out_file = os.path.join(out_dir, tag)
        logger.info(f"Clearing output file {out_file}")
        with open(out_file, "w") as _:
            pass

    return out_file, train, test, logger, prompter


def slike_setup(
    script:str,
    dataset:str, 
    model: str,
) -> tuple[MasterLogger, VisionPrompter]:
    logs_dir = f"./logs/{script}/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{dataset}.log")
    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=True,
    )

    PrompterCls: VisionPrompter = PROMPTER_CLASSES[model]
    prompter = PrompterCls()
    return logger, prompter
