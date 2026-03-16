"""
CUDA_VISIBLE_DECICES=0,1,2,3 python ./src/_download_mistral_random.py
"""

import json
import os
from pathlib import Path

import dotenv
from huggingface_hub import snapshot_download


MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
ENV_KEY = "MISTRAL_RANDOM_PROCESSOR_PATH"
ENV_FILE = Path("./resources/.env")


def patch_tokenizer_config(model_path: Path) -> None:
    """
    Ensure fix_mistral_regex=True is persisted ON DISK.
    This prevents duplicate kwarg injection when loading via AutoProcessor.
    """
    tok_cfg = model_path / "tokenizer_config.json"
    if not tok_cfg.exists():
        raise RuntimeError(f"tokenizer_config.json missing at {tok_cfg}")

    cfg = json.loads(tok_cfg.read_text())

    if cfg.get("fix_mistral_regex", False) is True:
        print("[OK] tokenizer_config.json already patched")
        return

    cfg["fix_mistral_regex"] = True
    tok_cfg.write_text(json.dumps(cfg, indent=2) + "\n")
    print("[PATCHED] Added fix_mistral_regex=True")


def main():
    # ------------------------------------------------------------
    # Download snapshot (processor + config only)
    # ------------------------------------------------------------
    path = Path(
        snapshot_download(
            MODEL_ID,
            allow_patterns=[
                "tokenizer*",
                "processor*",
                "*.json",
                "*.model",
            ],
            force_download=True,
        )
    )

    print(f"[OK] Snapshot at: {path}")

    # ------------------------------------------------------------
    # Patch tokenizer ONCE
    # ------------------------------------------------------------
    patch_tokenizer_config(path)

    # ------------------------------------------------------------
    # Persist path into .env
    # ------------------------------------------------------------
    dotenv.load_dotenv(ENV_FILE)

    if not ENV_FILE.exists():
        raise FileNotFoundError(f".env not found at {ENV_FILE}")

    lines = ENV_FILE.read_text().splitlines()
    lines = [l for l in lines if not l.startswith(f"{ENV_KEY}=")]
    lines.append(f'{ENV_KEY}="{path}"')

    ENV_FILE.write_text("\n".join(lines) + "\n")
    print(f"[OK] Saved {ENV_KEY} to {ENV_FILE}")


if __name__ == "__main__":
    main()
