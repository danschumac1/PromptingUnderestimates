from dataclasses import dataclass
import os
from typing import Any, Optional
from tqdm import tqdm

import numpy as np
from utils.random_prompter import RandomLlamaVisionPrompter, RandomMistralVisionPrompter, RandomQwenVisionPrompter
from utils.constants import (
    EXTRA_INFO_MAPPINGS, LEGEND_MAPPINGS, CoT_QUESTION_TAG, NO_CoT_QUESTION_TAG, 
    TASK_DESCRIPTION, TITLE_MAPPINGS, X_MAPPINGS, Y_MAPPINGS)
from utils.preprocessing import letcs_transform_multivar, serialize_ts
from utils.prompters import VisionPrompter
from utils.prompt_objects import VisionPrompt, LlamaVisPrompt, MistralVisPrompt, QwenVisPrompt
from utils.loaders import Split
from utils.visualization import plot_time_series

# -------------------------------------------
# HELPERS
# -------------------------------------------
MODEL_CONTEXT_LIMITS = {
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-3-large-v1": 8192,
    # "clip-ViT-L/14": None,  # not token-based
    # "ViT-L/14": None,
    # OPEN SOURCE
    "llama": 128000,
    "mistral": 128000,
    "qwen": 128000,
    "random_llama": 128000,
    "random_mistral": 128000,
    "random_qwen": 128000, 

}

PROMPT_DICT= {
    "llama": LlamaVisPrompt,
    "mistral": MistralVisPrompt,
    "qwen": QwenVisPrompt,
    "random_llama": LlamaVisPrompt,
    "random_mistral": MistralVisPrompt,
    "random_qwen": QwenVisPrompt

}

@dataclass
class RowHelper:
    idx: int
    X: np.ndarray
    class_id: int
    class_name: str
    class_letter: str

def build_row_helper_list(class_shots:dict[int:np.ndarray], label_maps:dict[str:dict[str:Any]], train:Split):
    '''
    Docstring for build_row_helper
    
    :param class_shots: dictionary of class_idx: list[int]
    '''
    list_row_helper:list[RowHelper] = []
    for class_id, exp_idx_list in class_shots.items():
        for exp_idx in exp_idx_list:
            row = RowHelper(
                idx = int(exp_idx),
                X = train[exp_idx].X,
                class_id=class_id,
                class_name=label_maps["id_to_name"][str(class_id)],
                class_letter=label_maps["id_to_letter"][str(class_id)]

            )
            list_row_helper.append(row)
    return list_row_helper


# -------------------------------------------
# SYSTEM PROMPT BUILDERS
# -------------------------------------------
def build_classification_system_prompt( # TODO
    dataset: str,
    CoT: bool = False,
) -> str:
    if CoT:
        system_prompt = (
            TASK_DESCRIPTION[dataset.upper()]
            + "\nYou will be given a multiple choice question and a time series. "
            "Your job is to use the time series to answer the question. "
            "Think step by step and explain your reasoning. "
            'At the end of your explanation, directly answer the question using exactly this format: '
            '"The answer is [X] CLASS_NAME". '
            'Example: "The answer is [D] CABBAGE".\n'
            + "Here is some additional information that may help you:\n"
            + EXTRA_INFO_MAPPINGS[dataset.upper()]
        )
    else:  # no chain of thought
        system_prompt = (
            TASK_DESCRIPTION[dataset.upper()]
            + "\nYou will be given a multiple choice question and a time series. "
            "Your job is to use the time series to answer the question. "
            "Do not explain your reasoning. "
            "Use exactly this format: "
            '"The answer is [X] CLASS_NAME". '
            'Example: "The answer is [D] CABBAGE".\n'
            + "Here is some additional information that may help you:\n"
            + EXTRA_INFO_MAPPINGS[dataset.upper()]
        )
  
    return system_prompt


def build_prompt(
    row: Split,
    split_name: str,
    *,
    dataset: str,
    model: str,
    user_text: str = "",
    include_ts: bool = False,
    include_LETSCLike: bool = False,
    include_vis: bool = False,
    assistant_msg: str = "",
) -> VisionPrompt:
    """
    Build a VisPrompt for a single classification example.

    - `include_ts`: append raw time series values as JSON to the user text.
    - `include_LETSCLike`: add LETSC-like digit-separated representation.
    - `include_vis`: render and attach an image of the time series.
    """
    if dataset.lower() == "rwc":
        viz_method = "spectrogram"
    elif dataset.lower() == "had":
        viz_method = "imu"
    else:
        viz_method = "line"     # Make sure we are not building a completely empty prompt
        
    assert include_ts or include_vis or include_LETSCLike, \
        "At least one of include_ts, include_vis, or include_LETSCLike must be True."
    # can only do include_ts OR include_LETSCLike; not both
    assert not (include_ts and include_LETSCLike), \
        "include_ts and include_LETSCLike cannot both be True."
    assert model in MODEL_CONTEXT_LIMITS, f"Model {model} not recognized."
    assert split_name in ["train", "test"], f"split_name {split_name} not recognized."

    if include_ts:
        ts_str = serialize_ts(row.X, MODEL_CONTEXT_LIMITS[model])
        if user_text:
            user_text += "\n\nHere are the raw time series values (JSON):\n" + ts_str
        else:
            user_text = ts_str

    if include_LETSCLike:
        ts_list = serialize_ts(row.X, MODEL_CONTEXT_LIMITS[model], return_as_list=True)
        letsclike_str = letcs_transform_multivar(ts_list)
        if user_text:
            user_text += "\n\nHere are the time series values formatted in a special style:\n" + letsclike_str
        else:
            user_text = letsclike_str

    image_path: Optional[str] = None
    if include_vis:
        img_dir = f"./data/images/{dataset}/{split_name}/"
        os.makedirs(img_dir, exist_ok=True)

        idx_scalar = int(np.asarray(row.idx).item())
        image_path = os.path.join(img_dir, f"{idx_scalar}.png")

        plot_time_series(
            row.X,
            method=viz_method,
            title=TITLE_MAPPINGS[dataset.upper()],
            xlabs=X_MAPPINGS[dataset.upper()],
            ylabs=Y_MAPPINGS[dataset.upper()],
            legends=LEGEND_MAPPINGS.get(dataset.upper(), None),
            save_path=image_path,
            recreate=True,
        )

    payload: dict = {}
    if user_text:
        payload["user_text"] = user_text
    if assistant_msg:
        payload["assistant_text"] = assistant_msg
    if image_path is not None:
        payload["image_path"] = image_path

    prompt = PROMPT_DICT[model](**payload)

    return prompt



def build_few_shot_classification_examples(
    class_shots: dict[str:np.ndarray],
    dataset: str,
    model: str,
    train: Split,
    test: Split,
    *,
    include_ts: bool = False,
    include_LETSCLike: bool = False,
    include_vis: bool = False,
    CoT: bool = False
) -> list[VisionPrompt]:
    """
    Build shared few-shot examples for classification, one per ex_idx in reasoning_dict.

    Returns:
        List[VisPrompt]: few-shot examples (used for every query).
    """
    row_helper_list = build_row_helper_list(class_shots, train.label_maps, train)
    # Shared question text with QUESTION_TAG
    general_question = (test.general_question or train.general_question).strip()
    assert general_question, "General question not available."
    if CoT:
        user_text_base = general_question + "\n\n" + CoT_QUESTION_TAG
    else:
        user_text_base = general_question + "\n\n" + NO_CoT_QUESTION_TAG

    few_shots: list[VisionPrompt] = []

    # deterministic ordering
    for row in row_helper_list:
        assistant_msg = \
            f"The answer is [{row.class_letter}] {row.class_name}"

        vp = build_prompt(
            row=train[row.idx],
            split_name="train",
            dataset=dataset,
            model=model,
            user_text=user_text_base,
            include_ts=include_ts,
            include_LETSCLike=include_LETSCLike,
            include_vis=include_vis,
            assistant_msg=assistant_msg,
        )
        few_shots.append(vp)
        
    return few_shots


def build_classification_query_prompts(
    batch_rows: Split,
    *,
    dataset: str,
    model: str,
    include_user_text: bool = True,
    include_ts: bool = False,
    include_LETSCLike: bool = False,
    include_vis: bool = True,
    CoT: bool = False
) -> list[VisionPrompt]:
    """
    Build query VisPrompts (no assistant_text) for a batch of test rows.
    """
    # Assume general_question is attached to the Split (like your loaders do)
    user_text_base = ""
    if include_user_text:
        general_question = batch_rows.general_question or batch_rows[0].general_question
        general_question = general_question.strip()
        if CoT:
            user_text_base = general_question + "\n\n" + CoT_QUESTION_TAG
        else: # no chain of thought
            user_text_base = general_question + "\n\n" + NO_CoT_QUESTION_TAG


    query_prompts: list[VisionPrompt] = []
    for row in batch_rows:
        vp = build_prompt(
            row=row,
            split_name="test",
            dataset=dataset,
            model=model,
            user_text=user_text_base,
            include_ts=include_ts,
            include_LETSCLike=include_LETSCLike,
            include_vis=include_vis,
        )
        query_prompts.append(vp)

    return query_prompts
        

