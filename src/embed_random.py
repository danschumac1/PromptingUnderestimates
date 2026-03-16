'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/embed_random.py 
'''

import argparse
import json

import torch

from utils.prompt_objects import LlamaVisPrompt
from utils.loaders import load_train_test
from utils.random_prompter import (
    # RandomLlamaVisionPrompter, 
    RandomMistralVisionPrompter, 
    # RandomQwenVisionPrompter
    )
from utils.prompters import (
    # LlamaVisionPrompter, 
    MistralVisionPrompter, 
    # QwenVisionPrompter
    )

model_map = {
    # "llama": (LlamaVisionPrompter, RandomLlamaVisionPrompter),
    "mistral": (MistralVisionPrompter, RandomMistralVisionPrompter),
    # "qwen": (QwenVisionPrompter, RandomQwenVisionPrompter)
}

MODEL = "mistral"

def main():
    # llama_prompter = LlamaVisionPrompter()
    regular_prompter_cls, random_prompter_cls = model_map[MODEL]
    regular_prompter = regular_prompter_cls()
    random_prompter = random_prompter_cls()

    # set system prompts
    random_prompter.system_prompt = "You are a helpful assistant"
    regular_prompter.system_prompt = "You are a helpful assistant"

    print('Generating llama embed...')
    llama_embed = regular_prompter.get_all_layer_embeddings([LlamaVisPrompt(user_text="What's the capital of France?")])
    print("\n\n")
    print('Generating random embed...')
    random_embed = random_prompter.get_all_layer_embeddings([LlamaVisPrompt(user_text="What's the capital of France?")])
    print(type(llama_embed), llama_embed.keys())
    print(type(random_embed), random_embed.keys())
    print(llama_embed['32'] == random_embed['32'])

if __name__ == "__main__":
    main()
