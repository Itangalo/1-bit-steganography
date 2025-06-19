from openai import OpenAI
import json
import re
import sys
import random
import asyncio

from config import (
    OPENROUTER_API_KEY, 
    DECODING_MODELS, 
    ENCODING_MODELS,
    NUM_ATTEMPTS, 
    NUM_BRANCHES, 
    MAX_TOKENS,
    CSV_DELIMITER
)

# OpenRouter setup
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Load prompts from files
def load_prompt(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Could not find {filename}")
        return None

ENCODING_USER_PROMPT = load_prompt('encoding_user_prompt.txt')
ENCODING_SYSTEM_PROMPT = load_prompt('encoding_system_prompt.txt')
DECODING_USER_PROMPT = load_prompt('decoding_user_prompt.txt')
DECODING_SYSTEM_PROMPT = load_prompt('decoding_system_prompt.txt')
VALID_SECRETS = ['Sweden', 'Denmark', 'Norway', 'Finland', 'Iceland', 'Germany', 'Poland', 'Czech Republic', 'Slovakia', 'Hungary', 'France', 'Spain', 'Italy', 'Portugal', 'Greece', 'United Kingdom', 'Ireland', 'Netherlands', 'Belgium', 'Luxembourg', 'Switzerland']


# Sync helper – wrapped in asyncio.to_thread later
def make_request(model: str, messages: list[dict[str, str]]) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"---Exception while encoding: {e}")
        return None

def extract_tag_from_response(text, tag):
    match = re.search(rf'<{tag}>\s*(.*?)\s*</{tag}>', text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None

def encode_message(encoding_model):
    chosen_secret = random.choice(VALID_SECRETS)
    prompt = [{"role": "system", "content": ENCODING_SYSTEM_PROMPT + chosen_secret}, {"role": "user", "content": ENCODING_USER_PROMPT + str(VALID_SECRETS)}]
    print(f"\n---Prompting the encoding model: {prompt}")
    raw_response = make_request(encoding_model, prompt)

    if not raw_response:
        print("---No response")
        return None

    print(f"\n---Raw response from encoding model:")
    print(raw_response)

    return {
        "secret": extract_tag_from_response(raw_response, "secret"),
        "reasoning": extract_tag_from_response(raw_response, "reasoning"), 
        "answer": extract_tag_from_response(raw_response, "answer")
    }

def output_decoding_result(attempt, encoding_model, decoding_model, encoding_model_response, decoding_model_response, branch, error, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{encoding_model}{CSV_DELIMITER}{attempt}{CSV_DELIMITER}{decoding_model}{CSV_DELIMITER}{branch}{CSV_DELIMITER}{encoding_model_response['answer']}{CSV_DELIMITER}{encoding_model_response['secret']}{CSV_DELIMITER}{decoding_model_response['solution'] if decoding_model_response else ''}{CSV_DELIMITER}{CSV_DELIMITER}{encoding_model_response['reasoning']}{CSV_DELIMITER}{decoding_model_response['reasoning'] if decoding_model_response else ''}{CSV_DELIMITER}{error}\n")

async def test_decoding_model(encoding_model_response, attempt, encoding_model, decoding_model, output_file):
    prompt = [{"role": "system", "content": DECODING_SYSTEM_PROMPT + str(VALID_SECRETS)}, {"role": "user", "content": DECODING_USER_PROMPT + "\n" + encoding_model_response['answer']}]
    sem = asyncio.Semaphore(NUM_BRANCHES)
    async_tasks = [
        asyncio.create_task(test_decoding_async(prompt, decoding_model))
        for branch in range(1, NUM_BRANCHES + 1)
    ]

    print(f"------ Starting decoding on {NUM_BRANCHES} branches.")
    decoding_model_responses = await asyncio.gather(*async_tasks)
    print(f"------ All decoded messages collected.")
    for branch, decoding_model_response in enumerate(decoding_model_responses, start=1):
        if decoding_model_response:
            print(f"------ Decoded message: {decoding_model_response}")
            output_decoding_result(attempt, encoding_model, decoding_model, encoding_model_response, decoding_model_response, branch, False, output_file)
        else:
            print(f"------ No response.")
            output_decoding_result(attempt, encoding_model, decoding_model, encoding_model_response, None, branch, True, output_file)

async def test_decoding_async(prompt, decoding_model):
    raw_response = make_request(decoding_model, prompt)
    print(f"\n------Raw response from decoding model:")
    print(raw_response)

    if not raw_response:
        print("---No response")
        return None

    return {
        "solution": extract_tag_from_response(raw_response, "solution"),
        "reasoning": extract_tag_from_response(raw_response, "reasoning")
    }

def get_output_file_name():
    if len(sys.argv) != 2:
        print("Usage: python steganography_test.py <CSV output filename>")
        sys.exit(1)

    return sys.argv[1]

async def main_async() -> None:
    output_file = get_output_file_name()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Encoding model{CSV_DELIMITER}Attempt{CSV_DELIMITER}Decoding model{CSV_DELIMITER}Branch{CSV_DELIMITER}Encoding answer{CSV_DELIMITER}Encoded secret{CSV_DELIMITER}Decoded solution{CSV_DELIMITER}Correct solution{CSV_DELIMITER}Encoding reasoning{CSV_DELIMITER}Decoding reasoning{CSV_DELIMITER}Correct reasoning{CSV_DELIMITER}Error\n")

    for encoding_model in ENCODING_MODELS:
        print(f"\n{'='*60}\n")
        print(f"\nTesting {encoding_model} as encoder")
        print(f"\n{'='*60}\n")

        for attempt in range(1, NUM_ATTEMPTS + 1):
            print(f"\n---Attempt {attempt}/{NUM_ATTEMPTS}")
            encoding_model_response = encode_message(encoding_model)

            if not encoding_model_response:
                print("---Encoding model failed; incorrectly formatted or missing response")
                continue
            else:
                for decoding_model in DECODING_MODELS:
                    print(f"\n------ Letting {decoding_model} decode")
                    await test_decoding_model(encoding_model_response, attempt, encoding_model, decoding_model, output_file)
 
        print(f"\n{'='*60}\n\n")

if __name__ == "__main__":
    asyncio.run(main_async())
