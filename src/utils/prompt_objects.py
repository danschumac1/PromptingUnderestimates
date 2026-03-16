from __future__ import annotations
from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


@dataclass
class VisionPrompt(ABC):
    image_path: Optional[str] = None
    user_text: Optional[str] = None
    assistant_text: Optional[str] = None

    # Standardized output for prompters to consume
    messages: list[dict] = field(init=False)

    def __post_init__(self) -> None:
        # shared validation
        if not (self.image_path or self.user_text):
            raise ValueError("At least one of image_path or user_text must be provided.")
        self.messages = self.build_messages()

    @abstractmethod
    def build_messages(self) -> list[dict]:
        """Return HF chat messages in the exact schema expected by the backend."""
        raise NotImplementedError


@dataclass
class TogetherVisPrompt(VisionPrompt):
    """
    Together follows OpenAI-style multimodal schema:

    messages = [{
      "role": "user",
      "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}}
      ]
    }]

    Notes:
      - Together does NOT accept local file paths as urls.
      - For local files, we embed as base64 data URLs.
    """

    image_mime: str = "image/jpeg"  # override if you need png etc.

    def _image_to_data_url(self, path: str) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{self.image_mime};base64,{b64}"

    def build_messages(self) -> list[dict]:
        content: list[dict] = []

        if self.user_text:
            content.append({"type": "text", "text": self.user_text})

        if self.image_path:
            url = self._image_to_data_url(self.image_path)
            content.append({"type": "image_url", "image_url": {"url": url}})

        msgs = [{"role": "user", "content": content}]

        if self.assistant_text:
            # Together/OpenAI expects assistant content to be text (or multimodal, but you likely want text)
            msgs.append({"role": "assistant", "content": self.assistant_text})

        return msgs


@dataclass
class LlamaVisPrompt(VisionPrompt):
    # Llama needs a PIL image loaded so the prompter can pass it separately.
    image: Optional[Image.Image] = field(init=False, default=None)

    def __post_init__(self) -> None:
        # load image (if present) before building messages
        if self.image_path:
            self.image = Image.open(self.image_path).convert("RGB")
        else:
            self.image = None
        super().__post_init__()

    def build_messages(self) -> list[dict]:
        content: list[dict] = []

        if self.image_path:
            # Placeholder only; image is passed separately via processor(images=...)
            content.append({"type": "image"})

        if self.user_text:
            content.append({"type": "text", "text": self.user_text})

        msgs = [{"role": "user", "content": content}]
        if self.assistant_text:
            msgs.append(
                {"role": "assistant", "content": [{"type": "text", "text": self.assistant_text}]}
            )
        return msgs

@dataclass
class MistralVisPrompt(VisionPrompt):
    """
    Same schema as Llama:
      - messages contain {"type":"image"} placeholder
      - actual PIL image is passed separately via processor(images=...)
    """
    image: Optional[Image.Image] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.image_path:
            self.image = Image.open(self.image_path).convert("RGB")
        else:
            self.image = None
        super().__post_init__()

    def build_messages(self) -> list[dict]:
        content: list[dict] = []

        if self.user_text:
            content.append({"type": "text", "text": self.user_text})

        if self.image_path:
            content.append({"type": "image"})  # placeholder

        msgs = [{"role": "user", "content": content}]
        if self.assistant_text:
            msgs.append(
                {"role": "assistant", "content": [{"type": "text", "text": self.assistant_text}]}
            )
        return msgs



@dataclass
class QwenVisPrompt(VisionPrompt):
    # Qwen does not require PIL image at prompt-build time
    # (process_vision_info will fetch/load images from the message schema)

    def build_messages(self) -> list[dict]:
        content: list[dict] = []

        if self.image_path:
            # Qwen wants the image reference INSIDE the message content
            content.append({"type": "image", "image": self.image_path})

        if self.user_text:
            content.append({"type": "text", "text": self.user_text})

        msgs = [{"role": "user", "content": content}]
        if self.assistant_text:
            msgs.append(
                {"role": "assistant", "content": [{"type": "text", "text": self.assistant_text}]}
            )
        return msgs
