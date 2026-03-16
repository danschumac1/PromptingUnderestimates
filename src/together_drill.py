'''python ./src/together_drill.py'''

import os
import dotenv

# must go before the import
os.environ["TOGETHER_NO_BANNER"] = "1"

# Your new objects
from utils.prompters import TogetherPrompter, GPT4Prompter
from utils.prompt_objects import TogetherVisPrompt

# PUBLIC VARIABLES
MODEL = "Qwen/Qwen3-VL-32B-Instruct"  # WORKS
# MODEL = "openai/gpt-oss-20b"  # WORKS
# MODEL = "nim/meta/llama-3.2-11b-vision-instruct"  # NEED DEDICATED ENDPOINT
# MODEL = "google/gemma-2-9b-it"  # NEED DEDICATED ENDPOINT
# MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"  # NOT AVAILABLE


def main():

    # TogetherPrompter will read TOGETHER_API_KEY from env by default
    prompter = TogetherPrompter(
        # model_id="gpt-4.1-mini",
        model_id=MODEL,
        system_prompt="Act as a sassy friend",      # optional
        temperature=0
    )

    prompts = [
        TogetherVisPrompt(
            image_path="demo/images/ex1.jpg",
            user_text="What is in this image?",
        )
    ]

    out = prompter.get_completion(prompts, batch=False)
    print(out)


if __name__ == "__main__":
    main()
