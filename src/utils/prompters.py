# STANDARD IMPORTS
from __future__ import annotations
from ctypes import Union
import os

import base64
import mimetypes
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar
# NEED PIP INSTALLS
from PIL import Image
import dotenv
os.environ["TOGETHER_NO_BANNER"] = "1"
from openai import OpenAI
from together import Together
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration, Mistral3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from huggingface_hub import login

# USER DEFINED
from .prompt_objects import TogetherVisPrompt, VisionPrompt, LlamaVisPrompt, MistralVisPrompt, QwenVisPrompt

#endregion
#region BASE CLASS CONSTRUCTION
# -------------------------------------------------------------------------------------------------
# BASE CLASS CONSTRUCTION
# -------------------------------------------------------------------------------------------------
P = TypeVar("P", bound=VisionPrompt)
class VisionPrompter(ABC, Generic[P]):
    """
    Parent class that exposes a unified API:
      - get_completion(prompts, batch=False)
      - get_embedding(prompts, batch=False, layer=-1)

    Children implement backend-specific hooks for:
      - creating serialized chat + vision inputs
      - processor call
      - trimming + decoding generated ids
    """

    def __init__(
        self,
        system_prompt: str = "",
        model_id: str = "",
        max_new_tokens: int = 2056,
        temperature: float = 0.1,
    ):
        self.system_prompt = system_prompt
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # children must set:
        #   self.model
        #   self.processor

    # -------------------------------
    # Shared helpers
    # -------------------------------
    def _warn_if_no_system(self, system_prompt: Optional[str], suppress_warnings: bool) -> None:
        if not suppress_warnings and not system_prompt:
            warnings.warn(
                "No system prompt provided; model behavior may be unpredictable.",
                UserWarning,
            )

    def _strip_assistant_header(self, text: str) -> str:
        stripped = text.lstrip()
        if stripped.lower().startswith("assistant"):
            parts = stripped.split("\n", 1)
            return parts[1].strip() if len(parts) == 2 else ""
        return text.strip()

    # -------------------------------
    # Backend hooks (must implement)
    # -------------------------------
    @abstractmethod
    def create_inputs(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ):
        """Return backend-specific objects describing one conversation (few-shot)."""
        raise NotImplementedError

    @abstractmethod
    def create_inputs_batch(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ):
        """Return backend-specific objects for true batching (one chat per prompt)."""
        raise NotImplementedError

    @abstractmethod
    def _processor_call_single(self, text, vision_inputs):
        """Return model inputs dict/tensors for single conversation."""
        raise NotImplementedError

    @abstractmethod
    def _processor_call_batch(self, texts, vision_inputs):
        """Return model inputs dict/tensors for batch."""
        raise NotImplementedError

    @abstractmethod
    def _trim_and_decode_single(self, inputs, generated_ids) -> str:
        """Return decoded completion string for single."""
        raise NotImplementedError

    @abstractmethod
    def _trim_and_decode_batch(self, inputs, generated_ids) -> list[str]:
        """Return decoded completion strings for batch."""
        raise NotImplementedError

    # -------------------------------
    # Unified public API: completion
    # -------------------------------
    def get_completion(self, prompts: list[VisionPrompt], batch: bool = False) -> str | list[str]:
        if batch:
            texts, vision_inputs = self.create_inputs_batch(prompts, system_prompt=self.system_prompt)
            inputs = self._processor_call_batch(texts, vision_inputs).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
            outs = self._trim_and_decode_batch(inputs, generated_ids)
            return [self._strip_assistant_header(x) for x in outs]

        text, vision_inputs = self.create_inputs(prompts, system_prompt=self.system_prompt)
        inputs = self._processor_call_single(text, vision_inputs).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        out = self._trim_and_decode_single(inputs, generated_ids)
        return self._strip_assistant_header(out)

    # -------------------------------
    # Unified public API: embedding
    # -------------------------------
    def get_embedding(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str] = None,
        batch: bool = False,
        layer: int = -1,
    ) -> torch.Tensor:
        sys = system_prompt or self.system_prompt

        if batch:
            texts, vision_inputs = self.create_inputs_batch(prompts, system_prompt=sys)
            inputs = self._processor_call_batch(texts, vision_inputs).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer]      # (B, L, D)
                print(f"hidden = outputs.hidden_states[layer] \t{hidden.shape}")
                embs = hidden[:, 0, :].float()             # (B, D)
                print(f"embs = hidden[:, 0, :].float() \t\t{embs.shape}")
                return embs

        text, vision_inputs = self.create_inputs(prompts, system_prompt=sys)
        inputs = self._processor_call_single(text, vision_inputs).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer]          # (1, L, D)
            print(f"hidden = outputs.hidden_states[layer] \t{hidden.shape}")
            embs = hidden[:, 0, :].squeeze(0).float()
            print(f"embs = hidden[:, 0, :].float() \t\t{embs.shape}")
            return embs       # (D,)

    # -------------------------------
    # Unified public API: all-layer embeddings
    # -------------------------------
    def get_all_layer_embeddings(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str] = None,
        batch: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Get embeddings from ALL layers using the LAST token position.
        
        Args:
            prompts: List of VisionPrompt objects.
            system_prompt: Optional system prompt override.
            batch: If True, process all prompts as a batch.
        
        Returns:
            Dict mapping layer index (as string) to embeddings tensor.
            Shape per layer: (B, D) if batch=True, else (D,).
        """
        sys = system_prompt or self.system_prompt

        if batch:
            texts, vision_inputs = self.create_inputs_batch(prompts, system_prompt=sys)
            inputs = self._processor_call_batch(texts, vision_inputs).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # outputs.hidden_states is tuple of (num_layers+1,) tensors of shape (B, L, D)
                all_layer_embs = {}
                for layer_idx, hidden in enumerate(outputs.hidden_states):
                    # Use LAST token position (index -1) for decoder-only models
                    embs = hidden[:, -1, :].float()  # (B, D)
                    all_layer_embs[str(layer_idx)] = embs
                return all_layer_embs

        text, vision_inputs = self.create_inputs(prompts, system_prompt=sys)
        inputs = self._processor_call_single(text, vision_inputs).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_layer_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                # Use LAST token position (index -1)
                embs = hidden[:, -1, :].squeeze(0).float()  # (D,)
                all_layer_embs[str(layer_idx)] = embs
            return all_layer_embs

#endregion
#region LLAMA PROMPTER
# -------------------------------------------------------------------------------------------------
# LLAMA PROMPTER
# -------------------------------------------------------------------------------------------------
class LlamaVisionPrompter(VisionPrompter[LlamaVisPrompt]):
    """
    NOTE: MLLama processor constraint:
      If you pass text as a batch (list[str]) and provide images,
      you must provide >=1 image per sample (no empty lists).
    We handle mixed batches by falling back to per-sample processing.
    """

    def __init__(
        self,
        system_prompt: str = "",
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        max_new_tokens: int = 2056,
        temperature: float = 0.1,
    ):
        super().__init__(system_prompt, model_id, max_new_tokens, temperature)

        dotenv.load_dotenv("./resources/.env")
        hf_token = os.environ["HF_TOKEN"]
        login(token=hf_token)


        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    # -------------------------------
    # Input builders
    # -------------------------------
    def create_inputs(
        self,
        prompts: list[LlamaVisPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> tuple[str, list[Image.Image]]:
        self._warn_if_no_system(system_prompt, suppress_warnings)

        chat: list[dict] = []
        images: list[Image.Image] = []

        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})

        for vp in prompts:
            chat.extend(vp.messages)
            if vp.image is not None:
                images.append(vp.image)

        text = self.processor.apply_chat_template(chat, add_generation_prompt=True)
        # print(f"INPUT_TEXT: type: {type(text)} | full text starting on next line\n {text}")
        return text, images

    def create_inputs_batch(
        self,
        prompts: list[LlamaVisPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> tuple[list[str], list[list[Image.Image]]]:
        self._warn_if_no_system(system_prompt, suppress_warnings)

        texts: list[str] = []
        images_nested: list[list[Image.Image]] = []

        for vp in prompts:
            chat: list[dict] = []
            if system_prompt:
                chat.append({"role": "system", "content": system_prompt})
            chat.extend(vp.messages)

            texts.append(self.processor.apply_chat_template(chat, add_generation_prompt=True))
            images_nested.append([vp.image] if vp.image is not None else [])
        # print(f"INPUT_TEXT: type: {type(texts)} | length: {len(texts)} | 0th element: {texts[0]}")

        return texts, images_nested

    # -------------------------------
    # Processor calls
    # -------------------------------
    def _processor_call_single(self, text: str, images: list[Image.Image]):
        kw = dict(text=text, add_special_tokens=False, return_tensors="pt")
        if images:
            kw["images"] = images
        inputs = self.processor(**kw)
        # # print(f"INPUTS | type: {type(inputs)} | length: {len(inputs)} | full: {inputs}")
        return inputs

    def _processor_call_batch(self, texts: list[str], images_nested: list[list[Image.Image]]):
        """
        IMPORTANT:
          If any images are provided, MLLama requires >=1 image per sample.
          We enforce that here (and rely on caller to fallback if mixed).
        """
        kw = dict(text=texts, add_special_tokens=False, padding=True, return_tensors="pt")

        has_any_images = any(len(x) > 0 for x in images_nested)
        if has_any_images:
            # must be 1+ image per sample
            if any(len(x) == 0 for x in images_nested):
                raise ValueError(
                    "MLLama batch requires either no images OR >=1 image per sample. "
                    "Got a mixed batch (some empty image lists)."
                )
            kw["images"] = images_nested

        inputs = self.processor(**kw)
        # print(f"INPUTS | type: {type(inputs)} | length: {len(inputs)} | full: {inputs}")
        return inputs

    # -------------------------------
    # Decode helpers
    # -------------------------------
    def _trim_and_decode_single(self, inputs, generated_ids) -> str:
        prompt_len = int(inputs["attention_mask"].sum(dim=1)[0].item())
        gen_ids = generated_ids[0, prompt_len:]
        return self.processor.decode(gen_ids, skip_special_tokens=True).strip()

    def _trim_and_decode_batch(self, inputs, generated_ids) -> list[str]:
        prompt_lens = inputs["attention_mask"].sum(dim=1)
        outs: list[str] = []
        for i in range(generated_ids.size(0)):
            plen = int(prompt_lens[i].item())
            gen_ids = generated_ids[i, plen:]
            outs.append(self.processor.decode(gen_ids, skip_special_tokens=True).strip())
        return outs

    # -------------------------------
    # Public API
    # -------------------------------
    def get_completion(self, prompts: list[LlamaVisPrompt], batch: bool = False) -> str | list[str]:
        if not batch:
            text, images = self.create_inputs(prompts, system_prompt=self.system_prompt)
            inputs = self._processor_call_single(text, images).to(self.model.device)

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )

            out = self._trim_and_decode_single(inputs, generated)
            return self._strip_assistant_header(out)

        # batch=True
        texts, images_nested = self.create_inputs_batch(prompts, system_prompt=self.system_prompt)

        # Detect mixed modality: some samples have images, some don't
        has_any_images = any(len(x) > 0 for x in images_nested)
        has_any_empty = any(len(x) == 0 for x in images_nested)

        if has_any_images and has_any_empty:
            # Fallback: run each sample as a single-item call
            outs: list[str] = []
            for p in prompts:
                outs.append(self.get_completion([p], batch=False))  # type: ignore[list-item]
            return outs

        # Safe to run as a true batch
        inputs = self._processor_call_batch(texts, images_nested).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        outs = self._trim_and_decode_batch(inputs, generated)
        return [self._strip_assistant_header(x) for x in outs]

    def get_embedding(
        self,
        prompts: list[LlamaVisPrompt],
        system_prompt: Optional[str] = None,
        batch: bool = False,
        layer: int = -1,
    ) -> torch.Tensor:
        sys = system_prompt or self.system_prompt

        if not batch:
            text, images = self.create_inputs(prompts, system_prompt=sys)
            inputs = self._processor_call_single(text, images).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer]  # (1, L, D)
                print(f"hidden = outputs.hidden_states[layer] \t{hidden.shape}")
                embs = hidden[:, 0, :].squeeze(0).float()  # (D,)
                print(f"embs = hidden[:, 0, :].float() \t\t{embs.shape}")
            return embs

        # batch=True
        texts, images_nested = self.create_inputs_batch(prompts, system_prompt=sys)

        has_any_images = any(len(x) > 0 for x in images_nested)
        has_any_empty = any(len(x) == 0 for x in images_nested)

        if has_any_images and has_any_empty:
            # Fallback: compute each embedding singly, then stack
            embs: list[torch.Tensor] = []
            for p in prompts:
                e = self.get_embedding([p], system_prompt=sys, batch=False, layer=layer)  # (D,)
                embs.append(e)
            return torch.stack(embs, dim=0)  # (B, D)

        inputs = self._processor_call_batch(texts, images_nested).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer]  # (B, L, D)
            print(f"hidden = outputs.hidden_states[layer] \t{hidden.shape}")
            embs = hidden[:, 0, :].float()         # (B, D)
            print(f"embs = hidden[:, 0, :].float() \t\t{embs.shape}")

        return embs

    def get_all_layer_embeddings(
        self,
        prompts: list[LlamaVisPrompt],
        system_prompt: Optional[str] = None,
        batch: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Get embeddings from ALL layers using the LAST token position.
        Handles mixed image/no-image batches by falling back to single processing.
        """
        sys = system_prompt or self.system_prompt

        if not batch:
            text, images = self.create_inputs(prompts, system_prompt=sys)
            inputs = self._processor_call_single(text, images).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)
                all_layer_embs = {}
                for layer_idx, hidden in enumerate(outputs.hidden_states):
                    embs = hidden[:, -1, :].squeeze(0).float()  # (D,)
                    all_layer_embs[str(layer_idx)] = embs
            return all_layer_embs

        # batch=True
        texts, images_nested = self.create_inputs_batch(prompts, system_prompt=sys)

        has_any_images = any(len(x) > 0 for x in images_nested)
        has_any_empty = any(len(x) == 0 for x in images_nested)

        if has_any_images and has_any_empty:
            # Fallback: compute each embedding singly, then merge dicts
            all_layer_embs: dict[str, list[torch.Tensor]] = {}
            for p in prompts:
                single_embs = self.get_all_layer_embeddings([p], system_prompt=sys, batch=False)
                for layer_key, emb in single_embs.items():
                    if layer_key not in all_layer_embs:
                        all_layer_embs[layer_key] = []
                    all_layer_embs[layer_key].append(emb)
            # Stack to (B, D) per layer
            return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

        inputs = self._processor_call_batch(texts, images_nested).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_layer_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                embs = hidden[:, -1, :].float()  # (B, D)
                all_layer_embs[str(layer_idx)] = embs

        return all_layer_embs

    def get_completions_and_embeddings(
            self, 
            prompts: list[LlamaVisPrompt], 
            batch: bool = False
        ) -> tuple[Union[str, list[str]], dict[str, torch.Tensor]]:
            """
            Unified call: Performs generation and captures the last layer 
            hidden states from the prompt in one pass.
            """
            sys = self.system_prompt
            
            if not batch:
                text, images = self.create_inputs(prompts, system_prompt=sys)
                inputs = self._processor_call_single(text, images).to(self.model.device)

                with torch.inference_mode():
                    # 1. Get Generation
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                    )
                    response = self._trim_and_decode_single(inputs, generated)
                    response = self._strip_assistant_header(response)

                    # 2. Get Last Layer Hidden State (using the model's forward pass)
                    # We only need the last token's representation of the prompt
                    outputs = self.model(**inputs, output_hidden_states=True)
                    last_layer_idx = len(outputs.hidden_states) - 1
                    
                    # Capture only the last layer at the last prompt token position
                    last_hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0).float() # (D,)
                    
                    # Return dictionary matching your combo script's expectations
                    return response, {str(last_layer_idx): last_hidden}

            # --- Batch logic ---
            texts, images_nested = self.create_inputs_batch(prompts, system_prompt=sys)
            has_any_images = any(len(x) > 0 for x in images_nested)
            has_any_empty = any(len(x) == 0 for x in images_nested)

            if has_any_images and has_any_empty:
                # Fallback for MLLama mixed modality constraint
                responses: list[str] = []
                layer_embs_list: list[torch.Tensor] = []
                final_key = ""

                for p in prompts:
                    resp, embs_dict = self.get_completions_and_embeddings([p], batch=False)
                    responses.append(resp)
                    # Assumes the dict only has one entry (the last layer)
                    final_key = list(embs_dict.keys())[0]
                    layer_embs_list.append(embs_dict[final_key])
                
                return responses, {final_key: torch.stack(layer_embs_list, dim=0)}

            # Standard Batch
            inputs = self._processor_call_batch(texts, images_nested).to(self.model.device)
            with torch.inference_mode():
                # Generate Text
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                responses = self._trim_and_decode_batch(inputs, generated)
                responses = [self._strip_assistant_header(x) for x in responses]

                # Get Hidden States
                outputs = self.model(**inputs, output_hidden_states=True)
                last_layer_idx = len(outputs.hidden_states) - 1
                last_hidden = outputs.hidden_states[-1][:, -1, :].float() # (B, D)

                return responses, {str(last_layer_idx): last_hidden}

#endregion
#region MISTRAL PROMPTER
# -------------------------------------------------------------------------------------------------
# MISTRAL PROMPTER
# -------------------------------------------------------------------------------------------------
P = TypeVar("P", bound="MistralVisPrompt")
class MistralVisionPrompter(VisionPrompter[P]):
    def __init__(
        self,
        system_prompt: str = "",
        model_id_or_path: Optional[str] = None,
        max_new_tokens: int = 2056,
        temperature: float = 0.15,
        local_files_only: bool = True,
        use_fast_processor: bool = False,   # match your scratchpad: False
        force_offline: bool = True,
        debug: bool = False,
    ):
        dotenv.load_dotenv("./resources/.env")

        if model_id_or_path is None:
            # model_id: str = mistralai/Mistral-Small-3.1-24B-Instruct-2503 # DON'T UNCOMMENT, JUST FOR REFERENCE
            model_id_or_path = os.environ["MISTRAL_SMALL_31_PATH"]

        if force_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        super().__init__(
            system_prompt=system_prompt,
            model_id=model_id_or_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        self.debug = debug

        self.processor = AutoProcessor.from_pretrained(
            model_id_or_path,
            local_files_only=local_files_only,
            use_fast=use_fast_processor,
        )

        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id_or_path,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=local_files_only,
        ).eval()


        # decoder-only => left padding in batch
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"

    # -------------------------------
    # Inputs
    # -------------------------------
    def create_inputs(
        self,
        prompts: list[MistralVisPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> tuple[str, list[Image.Image]]:
        self._warn_if_no_system(system_prompt, suppress_warnings)

        chat: list[dict] = []
        images: list[Image.Image] = []

        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})

        for vp in prompts:
            chat.extend(vp.messages)
            if getattr(vp, "image", None) is not None:
                images.append(vp.image)

        text = self.processor.apply_chat_template(chat, add_generation_prompt=True)
        # print(f"INPUT_TEXT: type: {type(text)} | {text}")

        return text, images

    def create_inputs_batch(
        self,
        prompts: list[MistralVisPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> tuple[list[str], list[list[Image.Image]]]:
        self._warn_if_no_system(system_prompt, suppress_warnings)

        texts: list[str] = []
        images_nested: list[list[Image.Image]] = []

        for vp in prompts:
            chat: list[dict] = []
            if system_prompt:
                chat.append({"role": "system", "content": system_prompt})
            chat.extend(vp.messages)

            texts.append(self.processor.apply_chat_template(chat, add_generation_prompt=True))
            images_nested.append([vp.image] if getattr(vp, "image", None) is not None else [])
        # print(f"INPUT_TEXT: type: {type(texts)} | length: {len(texts)} | 0th element: {texts[0]}")

        return texts, images_nested

    # -------------------------------
    # Processor calls
    # -------------------------------
    def _fix_vision_inputs(self, inputs):
        model_dtype = next(self.model.parameters()).dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        if "image_sizes" in inputs:
            inputs["image_sizes"] = inputs["image_sizes"].to(device=self.model.device)
        return inputs

    def _processor_call_single(self, text: str, images: list[Image.Image]):
        kw = dict(text=text, return_tensors="pt")
        if images:
            # single convo can have multiple images
            kw["images"] = images if len(images) > 1 else images[0]
        inputs = self.processor(**kw)
        # print(f"INPUTS | type: {type(inputs)} | length: {len(inputs)} | full: {inputs}")

        return self._fix_vision_inputs(inputs)

    def _processor_call_batch(self, texts: list[str], images_nested: list[list[Image.Image]]):
        # IMPORTANT: Mistral batches must be homogeneous (all have image OR none)
        has_img = [len(x) > 0 for x in images_nested]
        if any(has_img) and not all(has_img):
            raise ValueError(
                "Mixed text-only and image prompts in one batch. "
                "Split before calling _processor_call_batch."
            )

        kw = dict(text=texts, padding=True, return_tensors="pt")
        if all(has_img) and len(has_img) > 0:
            kw["images"] = [imgs[0] for imgs in images_nested]  # exactly one per sample

        inputs = self.processor(**kw)
        # print(f"INPUTS | type: {type(inputs)} | length: {len(inputs)} | full: {inputs}")

        return self._fix_vision_inputs(inputs)

    # -------------------------------
    # Trimming + decode (KEY FIX)
    # -------------------------------
    def _trim_and_decode_single(self, inputs, generated_ids) -> str:
        # Robust prompt length for Mistral3: use input_ids length (not attention_mask.sum)
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_ids = generated_ids[0, prompt_len:]

        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()

        if self.debug:
            print("\n[MISTRAL DEBUG single]")
            print("input_ids.shape:", tuple(inputs["input_ids"].shape))
            print("generated_ids.shape:", tuple(generated_ids.shape))
            print("prompt_len:", prompt_len)
            print("decoded(out):", repr(out[:300]))

        return out

    def _trim_and_decode_batch(self, inputs, generated_ids) -> list[str]:
        # Robust prompt length for Mistral3: use input_ids length for the batch
        prompt_len = int(inputs["input_ids"].shape[1])

        outs: list[str] = []
        for i in range(generated_ids.size(0)):
            gen_ids = generated_ids[i, prompt_len:]
            out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
            outs.append(out)

        if self.debug:
            print("\n[MISTRAL DEBUG batch]")
            print("input_ids.shape:", tuple(inputs["input_ids"].shape))
            print("generated_ids.shape:", tuple(generated_ids.shape))
            print("prompt_len:", prompt_len)
            for i, o in enumerate(outs):
                print(f"decoded[{i}] =", repr(o[:200]))

        return outs

    # -------------------------------
    # Batch splitting (keep it simple)
    # -------------------------------
    def _split_prompts(self, prompts: list[MistralVisPrompt]):
        text_idx, img_idx = [], []
        text_prompts, img_prompts = [], []
        for i, p in enumerate(prompts):
            if getattr(p, "image", None) is None:
                text_idx.append(i); text_prompts.append(p)
            else:
                img_idx.append(i); img_prompts.append(p)
        return (text_idx, text_prompts), (img_idx, img_prompts)

    def get_completion(self, prompts: list[MistralVisPrompt], batch: bool = False):
        if not batch:
            return super().get_completion(prompts, batch=False)

        (text_idx, text_prompts), (img_idx, img_prompts) = self._split_prompts(prompts)
        outs: list[Optional[str]] = [None] * len(prompts)

        if text_prompts:
            got = super().get_completion(text_prompts, batch=True)
            for i, s in zip(text_idx, got):
                outs[i] = s

        if img_prompts:
            got = super().get_completion(img_prompts, batch=True)
            for i, s in zip(img_idx, got):
                outs[i] = s

        return [o for o in outs if o is not None]

    def get_embedding(
        self,
        prompts: list[MistralVisPrompt],
        system_prompt: Optional[str] = None,
        batch: bool = False,
        layer: int = -1,
    ) -> torch.Tensor:
        if not batch:
            return super().get_embedding(prompts, system_prompt=system_prompt, batch=False, layer=layer)

        (text_idx, text_prompts), (img_idx, img_prompts) = self._split_prompts(prompts)
        rows: list[Optional[torch.Tensor]] = [None] * len(prompts)

        if text_prompts:
            e = super().get_embedding(text_prompts, system_prompt=system_prompt, batch=True, layer=layer)
            for r, i in enumerate(text_idx):
                rows[i] = e[r]

        if img_prompts:
            e = super().get_embedding(img_prompts, system_prompt=system_prompt, batch=True, layer=layer)
            for r, i in enumerate(img_idx):
                rows[i] = e[r]

        return torch.stack([r for r in rows if r is not None], dim=0)

    def get_all_layer_embeddings(
        self,
        prompts: list[MistralVisPrompt],
        system_prompt: Optional[str] = None,
        batch: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Get embeddings from ALL layers using the LAST token position.
        Handles text-only vs image prompts by splitting.
        """
        if not batch:
            return super().get_all_layer_embeddings(prompts, system_prompt=system_prompt, batch=False)

        (text_idx, text_prompts), (img_idx, img_prompts) = self._split_prompts(prompts)
        # Structure: {layer_key: [None] * len(prompts)}
        rows: dict[str, list[Optional[torch.Tensor]]] = {}

        if text_prompts:
            layer_embs = super().get_all_layer_embeddings(text_prompts, system_prompt=system_prompt, batch=True)
            for layer_key, embs in layer_embs.items():
                if layer_key not in rows:
                    rows[layer_key] = [None] * len(prompts)
                for r, i in enumerate(text_idx):
                    rows[layer_key][i] = embs[r]

        if img_prompts:
            layer_embs = super().get_all_layer_embeddings(img_prompts, system_prompt=system_prompt, batch=True)
            for layer_key, embs in layer_embs.items():
                if layer_key not in rows:
                    rows[layer_key] = [None] * len(prompts)
                for r, i in enumerate(img_idx):
                    rows[layer_key][i] = embs[r]

        # Stack per layer
        return {
            k: torch.stack([r for r in v if r is not None], dim=0)
            for k, v in rows.items()
        }


#endregion
#region QWEN PROMPTER
# -------------------------------------------------------------------------------------------------
# QWEN PROMPTER
# -------------------------------------------------------------------------------------------------
ImageInputs = Optional[list[Image.Image]]
class QwenVisionPrompter(VisionPrompter[QwenVisPrompt]):
    def __init__(
        self,
        system_prompt: str = "",
        model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        max_new_tokens: int = 2056,
        temperature: float = 0.1,
    ):
        super().__init__(system_prompt, model_id, max_new_tokens, temperature)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        # recommended: decoder-only left padding for batch
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

    def create_inputs(
        self,
        prompts: list[QwenVisPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> tuple[str, ImageInputs]:
        self._warn_if_no_system(system_prompt, suppress_warnings)

        chat: list[dict] = []
        if system_prompt:
            chat.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        for vp in prompts:
            chat.extend(vp.messages)

        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # print(f"INPUT_TEXT: type: {type(text)} | {text}")

        image_inputs, _ = process_vision_info(chat)
        return text, image_inputs

    def create_inputs_batch(
        self,
        prompts: list[QwenVisPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> tuple[list[str], ImageInputs]:
        sys = system_prompt or ""
        self._warn_if_no_system(sys, suppress_warnings)

        conversations: list[list[dict]] = []
        texts: list[str] = []

        for vp in prompts:
            chat: list[dict] = []
            if sys:
                chat.append({"role": "system", "content": [{"type": "text", "text": sys}]})
            chat.extend(vp.messages)
            conversations.append(chat)
            texts.append(self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

        image_inputs, _ = process_vision_info(conversations)
        # print(f"INPUT_TEXT: type: {type(texts)} | length: {len(texts)} | 0th element: {texts[0]}")

        return texts, image_inputs

    def _processor_call_single(self, text: str, image_inputs: ImageInputs):
        kw = dict(text=[text], padding=True, return_tensors="pt")
        if image_inputs is not None and len(image_inputs) > 0:
            kw["images"] = image_inputs
        inputs = self.processor(**kw)
        # print(f"INPUTS | type: {type(inputs)} | length: {len(inputs)} | full: {inputs}")
        return inputs

    def _processor_call_batch(self, texts: list[str], image_inputs: ImageInputs):
        kw = dict(text=texts, padding=True, return_tensors="pt")
        if image_inputs is not None and len(image_inputs) > 0:
            kw["images"] = image_inputs
        inputs = self.processor(**kw)
        # print(f"INPUTS | type: {type(inputs)} | length: {len(inputs)} | full: {inputs}")
        return inputs


    def _trim_and_decode_single(self, inputs, generated_ids) -> str:
        gen_trimmed = generated_ids[0, len(inputs.input_ids[0]):]
        return self.processor.batch_decode(
            [gen_trimmed],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _trim_and_decode_batch(self, inputs, generated_ids) -> list[str]:
        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )














class TogetherPrompter(VisionPrompter[TogetherVisPrompt]):
    """
    Drop-in prompter with the same public API shape as your HF prompters,
    but executes completions via Together’s hosted API.

    - get_completion(prompts, batch=False) -> str | list[str]
    - get_embedding(...) -> NotImplementedError
    - get_all_layer_embeddings(...) -> NotImplementedError
    """

    def __init__(
        self,
        model_id: str,
        system_prompt: str = "",
        max_new_tokens: int = 2056,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        env_path: Optional[str] = "./resources/.env",
        suppress_banner: bool = True,
    ):
        # Together banner must be disabled before importing Together in some setups,
        # but we can still set it here for safety.
        if suppress_banner:
            os.environ["TOGETHER_NO_BANNER"] = "1"

        super().__init__(
            system_prompt=system_prompt,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if env_path:
            dotenv.load_dotenv(env_path)

        key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise ValueError(
                "Together API key not found. Set TOGETHER_API_KEY or pass api_key=..."
            )

        self.client = Together(api_key=key)

    # -------------------------------
    # Together helpers
    # -------------------------------
    def _build_together_messages(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str],
        suppress_warnings: bool,
    ) -> list[dict]:
        """
        Combine few-shot prompts into one conversation list of messages.
        This mirrors your HF prompter behavior: prompts is a sequence of turns.
        """
        sys = system_prompt if system_prompt is not None else self.system_prompt
        self._warn_if_no_system(sys, suppress_warnings=suppress_warnings)

        messages: list[dict] = []
        if sys:
            messages.append({"role": "system", "content": sys})

        for p in prompts:
            # p.messages is already in backend schema by VisionPrompt design.
            # For TogetherVisPrompt, p.messages is a list of dicts.
            messages.extend(p.messages)

        return messages

    def _call_together(self, messages: list[dict]) -> str:
        messages = self._normalize_messages_for_together(messages)
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content


    # -------------------------------
    # Required hooks (kept for API parity)
    # -------------------------------
    def create_inputs(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ):
        messages = self._build_together_messages(prompts, system_prompt, suppress_warnings)
        return messages, None  # (text/messages, vision_inputs)

    def create_inputs_batch(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ):
        """
        “True batching” isn’t supported in the same way; we interpret batch=True
        as: each VisionPrompt in `prompts` is its own conversation.
        """
        # Here `prompts` is expected to be a list of *single* prompts (one conversation each).
        # If you want multi-turn conversations in batch mode, pass a list-of-lists instead
        # or build that at a higher layer.
        messages_list = [
            self._build_together_messages([p], system_prompt, suppress_warnings) for p in prompts
        ]
        return messages_list, None

    def _processor_call_single(self, text, vision_inputs):
        # Not used for Together (no torch tensors)
        return {"messages": text}

    def _processor_call_batch(self, texts, vision_inputs):
        return {"messages_list": texts}

    def _trim_and_decode_single(self, inputs, generated_ids) -> str:
        # Not used
        raise NotImplementedError

    def _trim_and_decode_batch(self, inputs, generated_ids) -> list[str]:
        # Not used
        raise NotImplementedError

    # -------------------------------
    # Override unified public API: completion
    # -------------------------------
    def get_completion(self, prompts: list[VisionPrompt], batch: bool = False) -> str | list[str]:
        if batch:
            messages_list, _ = self.create_inputs_batch(prompts, system_prompt=self.system_prompt)
            outs: list[str] = []
            for msgs in messages_list:
                out = self._call_together(msgs)
                outs.append(self._strip_assistant_header(out))
            return outs

        messages, _ = self.create_inputs(prompts, system_prompt=self.system_prompt)
        out = self._call_together(messages)
        return self._strip_assistant_header(out)

    # -------------------------------
    # Disable embedding APIs (as requested)
    # -------------------------------
    def get_embedding(self, *args, **kwargs):
        raise NotImplementedError("get_embedding is not implemented for TogetherPrompter.")

    def get_all_layer_embeddings(self, *args, **kwargs):
        raise NotImplementedError("get_all_layer_embeddings is not implemented for TogetherPrompter.")
    
    def _file_to_data_url(self, path: str) -> str:
        p = Path(path)
        mime, _ = mimetypes.guess_type(str(p))
        mime = mime or "image/png"
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"


    def _normalize_messages_for_together(self, messages: list[dict]) -> list[dict]:
        """
        Convert any multimodal parts shaped like:
        {"type": "image", "image": <path|data_url|{...}>}
        into Together-compatible:
        {"type": "image_url", "image_url": {"url": <data_url>}}
        """
        out: list[dict] = []
        for m in messages:
            m2 = dict(m)
            content = m2.get("content", None)

            # If content is a list of parts, normalize each part.
            if isinstance(content, list):
                new_parts: list[dict] = []
                for part in content:
                    if not isinstance(part, dict):
                        new_parts.append(part)
                        continue

                    ptype = part.get("type")

                    # Normalize {"type":"image", ...} -> {"type":"image_url", ...}
                    if ptype == "image":
                        img = part.get("image")
                        url: Optional[str] = None

                        if isinstance(img, str):
                            if img.startswith("data:"):
                                url = img
                            elif os.path.exists(img):
                                url = self._file_to_data_url(img)
                            else:
                                # some libs pass base64 directly; you can decide if you want to support it
                                raise ValueError(f"Unsupported image string content (not path or data url): {img[:50]}")
                        elif isinstance(img, dict):
                            # sometimes: {"url": "..."} or {"data": "base64..."}
                            if "url" in img and isinstance(img["url"], str):
                                url = img["url"]
                            elif "data" in img and isinstance(img["data"], str):
                                # assume raw base64
                                url = f"data:image/png;base64,{img['data']}"
                        # If your upstream uses a different key, add it here.

                        if not url:
                            raise ValueError(f"Could not normalize image payload: {part}")

                        new_parts.append({"type": "image_url", "image_url": {"url": url}})
                        continue

                    # pass through known-good types
                    if ptype in ("text", "image_url", "video_url", "audio_url"):
                        new_parts.append(part)
                        continue

                    # If unknown, keep it so we can see the failure
                    new_parts.append(part)

                m2["content"] = new_parts

            out.append(m2)

        return out



class GPT4Prompter(VisionPrompter[TogetherVisPrompt]):
    """
    OpenAI Chat Completions-backed prompter for gpt-4o-mini.

    Uses the same message schema you used for Together:
      {"role":"user","content":[{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:..."}}]}

    - get_completion(prompts, batch=False) -> str | list[str]
    - get_embedding(...) -> NotImplementedError
    - get_all_layer_embeddings(...) -> NotImplementedError
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        system_prompt: str = "",
        max_new_tokens: int = 2056,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        env_path: Optional[str] = "./resources/.env",
    ):
        super().__init__(
            system_prompt=system_prompt,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if env_path:
            dotenv.load_dotenv(env_path)

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=...")

        self.client = OpenAI(api_key=key)

    # -------------------------------
    # OpenAI helpers
    # -------------------------------
    def _build_openai_messages(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str],
        suppress_warnings: bool,
    ) -> list[dict]:
        sys = system_prompt if system_prompt is not None else self.system_prompt
        self._warn_if_no_system(sys, suppress_warnings=suppress_warnings)

        messages: list[dict] = []
        if sys:
            messages.append({"role": "system", "content": sys})

        for p in prompts:
            # TogetherVisPrompt already stores OpenAI-style messages in p.messages
            messages.extend(p.messages)

        return messages

    def _call_openai(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            # Chat Completions uses max_tokens (not max_new_tokens)
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    # -------------------------------
    # Required hooks (kept for API parity)
    # -------------------------------
    def create_inputs(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ):
        messages = self._build_openai_messages(prompts, system_prompt, suppress_warnings)
        return messages, None

    def create_inputs_batch(
        self,
        prompts: list[VisionPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ):
        # Batch=True is interpreted as: each prompt is its own conversation
        messages_list = [
            self._build_openai_messages([p], system_prompt, suppress_warnings) for p in prompts
        ]
        return messages_list, None

    def _processor_call_single(self, text, vision_inputs):
        return {"messages": text}

    def _processor_call_batch(self, texts, vision_inputs):
        return {"messages_list": texts}

    def _trim_and_decode_single(self, inputs, generated_ids) -> str:
        raise NotImplementedError

    def _trim_and_decode_batch(self, inputs, generated_ids) -> list[str]:
        raise NotImplementedError

    # -------------------------------
    # Override unified public API: completion
    # -------------------------------
    def get_completion(self, prompts: list[VisionPrompt], batch: bool = False) -> str | list[str]:
        if batch:
            messages_list, _ = self.create_inputs_batch(prompts, system_prompt=self.system_prompt)
            outs: list[str] = []
            for msgs in messages_list:
                out = self._call_openai(msgs)
                outs.append(self._strip_assistant_header(out))
            return outs

        messages, _ = self.create_inputs(prompts, system_prompt=self.system_prompt)
        out = self._call_openai(messages)
        return self._strip_assistant_header(out)

    # -------------------------------
    # Disable embedding APIs (same as Together version)
    # -------------------------------
    def get_embedding(self, *args, **kwargs):
        raise NotImplementedError("get_embedding is not implemented for GPT4oMiniPrompter.")

    def get_all_layer_embeddings(self, *args, **kwargs):
        raise NotImplementedError("get_all_layer_embeddings is not implemented for GPT4oMiniPrompter.")
