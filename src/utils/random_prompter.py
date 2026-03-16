import os
import time
from pathlib import Path
import re

import torch
from transformers import (
    AutoConfig, AutoProcessor, MllamaForConditionalGeneration, 
    Mistral3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)

from .prompters import (
    VisionPrompter, LlamaVisionPrompter, 
    MistralVisionPrompter, QwenVisionPrompter)

import dotenv
dotenv.load_dotenv("./resources/.env")


def _sanitize_repo_id(repo_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "__", repo_id)


class RandomLlamaVisionPrompter(LlamaVisionPrompter):
    """
    Same architecture + same processor as LlamaVisionPrompter,
    but model weights are randomly initialized once, then cached on disk.
    """

    def __init__(
        self,
        system_prompt: str = "",
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        max_new_tokens: int = 2056,
        temperature: float = 0.1,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/huggingface_hub/hub/random_init",
        force_reinit: int = 0,  # set to 1 to rebuild even if exists
    ):
        print("Initializing RandomVisionPrompter (load-or-random-init)...")

        # IMPORTANT: call VisionPrompter init, not LlamaVisionPrompter init
        VisionPrompter.__init__(
            self,
            system_prompt=system_prompt,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Where we store the cached random model
        dtype_tag = "bf16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
        save_dir = Path(cache_root) / _sanitize_repo_id(model_id) / f"dtype_{dtype_tag}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Always use the same processor as the pretrained model (or cached copy)
        # Prefer local copy if it exists, otherwise download from model_id and later save.
        processor_src = str(save_dir) if (save_dir / "preprocessor_config.json").exists() else model_id

        print(f"Processor source: {processor_src}")
        self.processor = AutoProcessor.from_pretrained(processor_src)

        model_files_exist = (save_dir / "config.json").exists() and (
            (save_dir / "model.safetensors").exists()
            or any(save_dir.glob("model-*.safetensors"))
            or (save_dir / "pytorch_model.bin").exists()
            or any(save_dir.glob("pytorch_model-*.bin"))
        )

        if model_files_exist and not force_reinit:
            print(f"[cache] Found cached random model at: {save_dir}")
            t = time.time()
            self.model = MllamaForConditionalGeneration.from_pretrained(
                str(save_dir),
                torch_dtype=dtype,
                device_map=None,     # keep explicit device placement
            ).to(device).eval()
            print(f"[cache] Loaded in {time.time()-t:.1f}s on {device}")
            return

        # ------------------------------------------------------------------
        # Build random model on GPU (Option B), then save for future runs
        # ------------------------------------------------------------------
        print(f"[build] No cached model (or force_reinit=1). Building random weights on {device}...")

        print("[build] Loading config...")
        config = AutoConfig.from_pretrained(model_id)

        # Initialize directly on device in requested dtype
        old_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            t = time.time()
            model = MllamaForConditionalGeneration(config)
            # Move immediately to device (avoid CPU->GPU double-touch later)
            model = model.to(device).eval()
            print(f"[build] Constructed + moved to {device} in {time.time()-t:.1f}s")
        finally:
            torch.set_default_dtype(old_default)

        self.model = model

        # Save to disk (move to CPU first to avoid GPU spikes / weirdness)
        print(f"[save] Saving random model to: {save_dir}")
        t = time.time()
        cpu_model = self.model.to("cpu")
        cpu_model.save_pretrained(str(save_dir), safe_serialization=True)
        self.processor.save_pretrained(str(save_dir))
        print(f"[save] Done in {time.time()-t:.1f}s")

        # Put it back where the caller asked (optional, but nice)
        if device != "cpu":
            self.model = cpu_model.to(device).eval()
        else:
            self.model = cpu_model.eval()


# class RandomMistralVisionPrompter(MistralVisionPrompter):
#     def __init__(
#         self,
#         system_prompt: str = "",
#         device: str | None = None,
#         dtype: torch.dtype = torch.bfloat16,
#         debug: bool = False, # 1. Add debug to arguments
#     ):
#         # --------------------------------------------------
#         # IMPORTANT: skip MistralVisionPrompter.__init__
#         # --------------------------------------------------
#         VisionPrompter.__init__(
#             self,
#             system_prompt=system_prompt,
#             model_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
#         )
#         self.debug = debug

#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#         # --------------------------------------------------
#         # Load patched processor only
#         # --------------------------------------------------
#         self.processor = AutoProcessor.from_pretrained(
#             os.environ["MISTRAL_RANDOM_PROCESSOR_PATH"],
#             local_files_only=True,
#             use_fast=False,
#         )

#         if hasattr(self.processor, "tokenizer"):
#             self.processor.tokenizer.padding_side = "left"

#         # --------------------------------------------------
#         # Random model
#         # --------------------------------------------------
#         config = AutoConfig.from_pretrained(
#             "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
#         )

#         old = torch.get_default_dtype()
#         torch.set_default_dtype(dtype)
#         try:
#             self.model = Mistral3ForConditionalGeneration(config)
#         finally:
#             torch.set_default_dtype(old)

#         self.model = self.model.to(device).eval()

class RandomMistralVisionPrompter(MistralVisionPrompter):
    def __init__(
        self,
        system_prompt: str = "",
        model_id: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/huggingface_hub/hub/random_init",
        force_reinit: int = 0,
        debug: bool = False,
    ):
        # 1. Initialize base class (skipping MistralVisionPrompter's weight loading)
        VisionPrompter.__init__(
            self,
            system_prompt=system_prompt,
            model_id=model_id,
        )
        self.debug = debug
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2. Setup caching paths
        dtype_tag = "bf16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
        save_dir = Path(cache_root) / _sanitize_repo_id(model_id) / f"dtype_{dtype_tag}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 3. Load the patched processor
        # Note: We use the ENV variable for the specialized Mistral processor
        self.processor = AutoProcessor.from_pretrained(
            os.environ["MISTRAL_RANDOM_PROCESSOR_PATH"],
            local_files_only=True,
            use_fast=False,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        # 4. Check for cached weights
        model_files_exist = (save_dir / "config.json").exists() and (
            (save_dir / "model.safetensors").exists() 
            or any(save_dir.glob("model-*.safetensors"))
        )

        if model_files_exist and not force_reinit:
            print(f"[cache] Found cached random Mistral at: {save_dir}")
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                str(save_dir),
                torch_dtype=dtype,
                device_map=None,
            ).to(device).eval()
            return

        # 5. Build random weights if not cached
        print(f"[build] Building random Mistral-Small weights on {device}...")
        config = AutoConfig.from_pretrained(model_id)
        
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            # We initialize directly on device to prevent OOM on system RAM
            with torch.device(device):
                self.model = Mistral3ForConditionalGeneration(config)
        finally:
            torch.set_default_dtype(old_dtype)

        # 6. Save to disk for future consistency
        print(f"[save] Persisting random weights to: {save_dir}")
        # Move to CPU for saving to avoid potential GPU-to-disk hang/instability
        cpu_model = self.model.to("cpu")
        cpu_model.save_pretrained(str(save_dir), safe_serialization=True)
        self.processor.save_pretrained(str(save_dir))
        
        # Move back to target device
        self.model = cpu_model.to(device).eval()


class RandomQwenVisionPrompter(QwenVisionPrompter):
    """
    Randomly initialized Qwen2.5-VL vision model with cached weights.
    """

    def __init__(
        self,
        system_prompt: str = "",
        model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        max_new_tokens: int = 2056,
        temperature: float = 0.1,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/huggingface_hub/hub/random_init",
        force_reinit: int = 0,
    ):
        print("Initializing RandomQwenVisionPrompter (load-or-random-init)...")

        # IMPORTANT: bypass QwenVisionPrompter.__init__
        VisionPrompter.__init__(
            self,
            system_prompt=system_prompt,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_tag = "bf16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
        save_dir = Path(cache_root) / _sanitize_repo_id(model_id) / f"dtype_{dtype_tag}"
        save_dir.mkdir(parents=True, exist_ok=True)

        processor_src = str(save_dir) if (save_dir / "preprocessor_config.json").exists() else model_id
        print(f"Processor source: {processor_src}")
        self.processor = AutoProcessor.from_pretrained(processor_src)

        # left padding
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        model_files_exist = (
            (save_dir / "config.json").exists()
            and (
                (save_dir / "model.safetensors").exists()
                or any(save_dir.glob("model-*.safetensors"))
                or (save_dir / "pytorch_model.bin").exists()
                or any(save_dir.glob("pytorch_model-*.bin"))
            )
        )

        if model_files_exist and not force_reinit:
            print(f"[cache] Found cached random model at: {save_dir}")
            t = time.time()
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(save_dir),
                torch_dtype=dtype,
                device_map=None,
            ).to(device).eval()
            print(f"[cache] Loaded in {time.time()-t:.1f}s on {device}")
            return

        # ------------------------------------------------------------
        # Build random model
        # ------------------------------------------------------------
        print(f"[build] Building random Qwen2.5-VL on {device}...")

        config = AutoConfig.from_pretrained(model_id)

        old_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            t = time.time()
            model = Qwen2_5_VLForConditionalGeneration(config)
            model = model.to(device).eval()
            print(f"[build] Constructed + moved in {time.time()-t:.1f}s")
        finally:
            torch.set_default_dtype(old_default)

        self.model = model

        print(f"[save] Saving random model to: {save_dir}")
        t = time.time()
        cpu_model = self.model.to("cpu")
        cpu_model.save_pretrained(str(save_dir), safe_serialization=True)
        self.processor.save_pretrained(str(save_dir))
        print(f"[save] Done in {time.time()-t:.1f}s")

        self.model = cpu_model.to(device).eval() if device != "cpu" else cpu_model.eval()


