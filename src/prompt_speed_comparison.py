import os
import requests
import json
import re
from dotenv import load_dotenv
import time
import concurrent.futures

# Load environment variables from a .env file
load_dotenv()

# Get the API key from environment variables
API_KEY = os.getenv("AIML_API_KEY")
API_URL = "https://api.aimlapi.com/v1/chat/completions"

# --- Function Definitions (copied from documentary_generator.py) ---

def split_into_phrases(text):
    """
    Splits the narration script into a list of short phrases (3-5 seconds).
    """
    words = text.replace('\n', ' ').split()
    phrases = []
    current_phrase = ""
    for word in words:
        if not current_phrase:
            current_phrase = word
        else:
            current_phrase += " " + word
        if len(current_phrase.split()) >= 8 and word.endswith(('.', '!', '?', ',')):
            phrases.append(current_phrase)
            current_phrase = ""
        elif len(current_phrase.split()) >= 15:
            phrases.append(current_phrase)
            current_phrase = ""
    if current_phrase:
        phrases.append(current_phrase)
    return phrases

# --- Implementation 1: Sequential (Old Method) ---

def generate_flux_prompts_sequential(phrases):
    """
    Generates prompts sequentially, one after another.
    """
    prompts = []
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    for i, phrase in enumerate(phrases):
        prompt_for_flux = f"Based on the narration phrase, create a visual prompt for a text-to-image model. Narration Phrase: \"{phrase}\""
        data = { "model": "gpt-4o", "messages": [{"role": "user", "content": prompt_for_flux}], "max_tokens": 4096 }
        try:
            response = requests.post(API_URL, headers=headers, json=data)
            response.raise_for_status()
            prompts.append(response.json()['choices'][0]['message']['content'].strip())
        except requests.exceptions.RequestException as e:
            print(f"Error for phrase {i+1}: {e}")
            prompts.append(None)
    return prompts

# --- Implementation 2: Concurrent (New Method) ---

def _generate_single_prompt(phrase_with_index):
    """
    Helper function to generate a single prompt. Runs in a separate thread.
    """
    i, phrase = phrase_with_index
    prompt_for_flux = f"Based on the narration phrase, create a visual prompt for a text-to-image model. Narration Phrase: \"{phrase}\""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt_for_flux}], "max_tokens": 4096}
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        flux_prompt = response.json()['choices'][0]['message']['content'].strip()
        return (i, flux_prompt)
    except requests.exceptions.RequestException as e:
        print(f"Error for phrase {i+1}: {e}")
        return (i, None)

def generate_flux_prompts_concurrent(phrases):
    """
    Generates prompts concurrently using a thread pool.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        phrases_with_indices = list(enumerate(phrases))
        results = list(executor.map(_generate_single_prompt, phrases_with_indices))
    results.sort(key=lambda x: x[0])
    return [prompt for i, prompt in results]

# --- Main execution block ---

if __name__ == "__main__":
    sample_text = (
        "The ancient ruins stood silent under the watchful eyes of the moon. "
        "For centuries, they had guarded a secret, a story lost to time. "
        "Legends whispered of a hidden chamber, a place of immense power. "
        "But no one had ever found it, until now. A young archaeologist, "
        "driven by a thirst for knowledge, discovered a hidden map. The map was old, "
        "its parchment brittle with age. It pointed to a location deep within the jungle, "
        "a place no one dared to venture. This is where her journey begins."
    )

    phrases_to_test = split_into_phrases(sample_text)
    num_prompts = len(phrases_to_test)
    print(f"--- Starting Prompt Generation Speed Test ({num_prompts} prompts) ---\n")

    # Test Sequential
    print("1. Testing Sequential (Old) Implementation...")
    start_seq = time.time()
    generate_flux_prompts_sequential(phrases_to_test)
    end_seq = time.time()
    total_seq = end_seq - start_seq
    avg_seq = total_seq / num_prompts
    print(f"Sequential implementation finished in {total_seq:.2f} seconds.")
    print(f"Average time per prompt: {avg_seq:.2f} seconds.\n")

    # Test Concurrent
    print("2. Testing Concurrent (New) Implementation...")
    start_con = time.time()
    generate_flux_prompts_concurrent(phrases_to_test)
    end_con = time.time()
    total_con = end_con - start_con
    avg_con = total_con / num_prompts
    print(f"Concurrent implementation finished in {total_con:.2f} seconds.")
    # Note: Average time here is misleading for concurrent execution, but we calculate it for comparison.
    print(f"Effective average time per prompt: {avg_con:.2f} seconds.\n")

    # Summary
    print("--- Performance Summary ---")
    print(f"Total time for {num_prompts} prompts (Sequential): {total_seq:.2f}s")
    print(f"Total time for {num_prompts} prompts (Concurrent): {total_con:.2f}s")
    speed_improvement = (total_seq / total_con) if total_con > 0 else float('inf')
    print(f"\nThe concurrent implementation was {speed_improvement:.2f} times faster.")
