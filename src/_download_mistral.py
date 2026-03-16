"""
CUDA_VISIBLE_DECIVES=0,1,2,3 python ./src/_download_mistral.py
"""

import json
import os
from pathlib import Path

import dotenv
from huggingface_hub import snapshot_download


MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
ENV_KEY = "MISTRAL_SMALL_31_PATH"
ENV_FILE = Path("./resources/.env")


def patch_tokenizer_config(model_path: Path) -> None:
    """
    Ensure fix_mistral_regex=True is set in tokenizer_config.json.
    This avoids passing fix_mistral_regex twice via kwargs and
    works around a Transformers kwarg collision bug.
    """
    tok_cfg_path = model_path / "tokenizer_config.json"

    if not tok_cfg_path.exists():
        print(f"[WARN] tokenizer_config.json not found at {tok_cfg_path}")
        return

    cfg = json.loads(tok_cfg_path.read_text())

    if cfg.get("fix_mistral_regex", False) is True:
        print("[OK] tokenizer_config.json already patched")
        return

    cfg["fix_mistral_regex"] = True
    tok_cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    print("[PATCHED] Added fix_mistral_regex=True to tokenizer_config.json")


def main():
    # ------------------------------------------------------------------
    # Download snapshot (idempotent)
    # ------------------------------------------------------------------
    path = Path(snapshot_download(MODEL_ID, force_download=True))
    print(f"[OK] Model snapshot at: {path}")

    # ------------------------------------------------------------------
    # Patch tokenizer config ONCE
    # ------------------------------------------------------------------
    patch_tokenizer_config(path)

    # ------------------------------------------------------------------
    # Persist path to .env
    # ------------------------------------------------------------------
    dotenv.load_dotenv(ENV_FILE)

    if not ENV_FILE.exists():
        raise ValueError(
            f".env file not found at {ENV_FILE}. "
            "Please create it first."
        )

    lines = ENV_FILE.read_text().splitlines()
    lines = [l for l in lines if not l.startswith(f"{ENV_KEY}=")]
    lines.append(f'{ENV_KEY}="{path}"')

    ENV_FILE.write_text("\n".join(lines) + "\n")

    print(f"[OK] Saved {ENV_KEY} to {ENV_FILE}")
    print(path)


if __name__ == "__main__":
    main()
