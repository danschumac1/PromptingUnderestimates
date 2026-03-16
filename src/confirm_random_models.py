'''
How to run:
   python ./src/confirm_random_models.py
'''

from utils.prompt_objects import LlamaVisPrompt
from utils.random_prompter import (
    VisionPrompter, RandomLlamaVisionPrompter, 
    RandomMistralVisionPrompter, RandomQwenVisionPrompter
    )

RANDOM_MODEL_MAPPING ={
    # "llama": RandomLlamaVisionPrompter,
    "mistral": RandomMistralVisionPrompter,
    # "qwen": RandomQwenVisionPrompter
}

MODEL = "mistral"

def main():
    prompter = RANDOM_MODEL_MAPPING[MODEL]()
    prompter.system_prompt = "You are a helpful assistant."
    print("Prompter initialized successfully.")
    output = prompter.get_completion([LlamaVisPrompt(user_text="What is the capital of france?")])
    print("Output: ", output)

if __name__ == "__main__":
    main()