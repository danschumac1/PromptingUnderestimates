"""
Generate multiple meaning-preserving variations of:
1) a classification SYSTEM PROMPT
2) the dataset GENERAL QUESTION

Batched generation: if --num_variants=10 and --batch_size=5, we do 2 calls of 5.

Example:
python src/generate_prompt_variations.py \
    --dataset emg \
    --num_variants 10 \
    --batch_size 5
"""

import argparse
import json
import os
import re
from typing import List, Dict, Any

from utils.build_prompts import build_classification_system_prompt
from utils.prompt_objects import TogetherVisPrompt
from utils.constants import build_valid_embedding_strings
from utils.prompters import GPT4Prompter
from utils.loaders import load_train_test


# -------------------------
# Argument parsing
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--normalize", type=int, choices=[0, 1], default=1)
    p.add_argument("--num_variants", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=5)
    return p.parse_args()


# -------------------------
# Prompts
# -------------------------
PROMPT_VARIATION_SYSTEM_PROMPT = """
You are a prompt rewriter.

Rewrite the provided SYSTEM PROMPT into N distinct SYSTEM PROMPT variants that
preserve meaning, constraints, and all factual task content, while changing
phrasing, structure, and formatting.

HARD CONSTRAINTS (must obey):
- Do NOT add, remove, rename, or reorder any class names.
- Do NOT alter the semantic meaning of any class description.
- Preserve any answer-choice letters (e.g., [A], [B], etc.) exactly and in order.
- Preserve required output format strings exactly, including punctuation and casing,
  e.g. "The answer is [X] CLASS_NAME".
- Do NOT introduce new task instructions (e.g., explanations, reasoning, confidence).
- Maintain an academically appropriate tone.

DIVERSITY REQUIREMENTS:
- Each variant must differ meaningfully in organization and wording.
- Use multiple presentation styles across variants (headings, bullet lists,
  numbered steps, compact rubrics, etc.).
- Avoid trivial paraphrases.

OUTPUT FORMAT (strict):
Return valid JSON ONLY with the following structure:

{
  "variants": [
    {"id": 1, "system_prompt": "..."},
    {"id": 2, "system_prompt": "..."}
  ]
}

No extra keys. No markdown. No commentary.
""".strip()

GENERAL_QUESTION_VARIATION_SYSTEM_PROMPT = """
You are a question rewriter.

Your task is to rewrite a GENERAL QUESTION into N distinct variants that preserve
the task, label space, and decision criteria, while changing wording, structure,
and phrasing.

HARD CONSTRAINTS (must obey):
- Do NOT change, rename, remove, or reorder answer choices.
- Preserve answer-choice letters exactly (e.g., [A], [B]) and their order.
- Do NOT introduce new labels, hints, or constraints.
- Do NOT add explanations, reasoning instructions, or output formatting rules.
- The rewritten question must ask the same classification decision.

DIVERSITY REQUIREMENTS:
- Each variant must differ meaningfully in wording and structure.
- Vary tone (instructional vs role-based), sentence structure, and framing.
- Avoid trivial paraphrases.

OUTPUT FORMAT (strict):
Return valid JSON ONLY with the following structure:

{
  "variants": [
    {"id": 1, "question": "..."},
    {"id": 2, "question": "..."}
  ]
}

No extra keys. No markdown. No commentary.
""".strip()


# -------------------------
# Helpers
# -------------------------

def parse_variants_json(raw_output: str, expected_key: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(raw_output)
        variants = parsed["variants"]
        if not isinstance(variants, list):
            raise ValueError("'variants' is not a list")
        for i, v in enumerate(variants):
            if expected_key not in v:
                raise KeyError(f"Missing key '{expected_key}' in variants[{i}]")
        return variants
    except Exception as e:
        raise RuntimeError(f"Model did not return valid JSON. Raw output:\n{raw_output}") from e


def generate_variants_in_batches(
    *,
    prompter: GPT4Prompter,
    base_text: str,
    user_instruction: str,
    expected_key: str,
    total: int,
    batch_size: int,
    validate_fn=None,
) -> List[Dict[str, Any]]:
    if total <= 0:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if total % batch_size != 0:
        raise ValueError(f"num_variants ({total}) must be divisible by batch_size ({batch_size})")

    all_variants: List[Dict[str, Any]] = []
    num_batches = total // batch_size

    for batch_idx in range(num_batches):
        messages = [
            TogetherVisPrompt(
                user_text=(
                    f"Generate {batch_size} rewritten variants of the following.\n\n"
                    f"{user_instruction}\n\n"
                    f"{base_text}"
                )
            )
        ]

        raw_output = prompter.get_completion(messages)
        batch_variants = parse_variants_json(raw_output, expected_key=expected_key)

        # Renumber globally & validate
        for i, v in enumerate(batch_variants):
            global_id = batch_idx * batch_size + i + 1
            v["id"] = global_id

            if validate_fn is not None:
                validate_fn(v)

            all_variants.append(v)

    return all_variants


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    base_dir = f"./data/features/prompts/{args.dataset}"
    system_dir = os.path.join(base_dir, "system")
    gq_dir = os.path.join(base_dir, "gq")

    os.makedirs(system_dir, exist_ok=True)
    os.makedirs(gq_dir, exist_ok=True)


    # Load dataset (only needed for test.general_question here)
    _, test = load_train_test(
        os.path.join("./data/samples", args.dataset),
        0,
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )

    # Build original system prompt
    original_system_prompt = build_classification_system_prompt(args.dataset)

    
    # Initialize prompter
    prompter = GPT4Prompter(
        model_id="gpt-4.1-mini",
        temperature=0.6,
    )

    # -------------------------
    # 1) SYSTEM PROMPT VARIATIONS (batched)
    # -------------------------
    prompter.system_prompt = PROMPT_VARIATION_SYSTEM_PROMPT

    sp_variants = generate_variants_in_batches(
        prompter=prompter,
        base_text=original_system_prompt,
        user_instruction=(
            "IMPORTANT:\n"
            "- Do NOT change class names.\n"
            "- Do NOT change answer letters or their order.\n"
            "- Do NOT change the required answer format.\n"
            "- Output must be valid JSON only."
        ),
        expected_key="system_prompt",
        total=args.num_variants,
        batch_size=args.batch_size,
        validate_fn=None,  # you can add checks here if you want
    )

    for v in sp_variants:
        path = os.path.join(system_dir, f"{v['id']}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(v["system_prompt"].strip() + "\n")


    print("\nSYSTEM PROMPT VARIATIONS DONE ✅")

    # -------------------------
    # 2) GENERAL QUESTION VARIATIONS (batched)
    # -------------------------
    prompter.system_prompt = GENERAL_QUESTION_VARIATION_SYSTEM_PROMPT


    q_variants = generate_variants_in_batches(
        prompter=prompter,
        base_text=test.general_question,
        user_instruction=(
            "IMPORTANT:\n"
            "- Do NOT change the answer choice letters, names, or their order.\n"
            "- Do NOT add output formatting rules.\n"
            "- Do NOT add reasoning/explanation instructions.\n"
            "- Output must be valid JSON only."
        ),
        expected_key="question",
        total=args.num_variants,
        batch_size=args.batch_size,
    )

    for v in q_variants:
        path = os.path.join(gq_dir, f"{v['id']}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(v["question"].strip() + "\n")


    print("\nGENERAL QUESTION VARIATIONS DONE ✅✅✅")
