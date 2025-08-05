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


def generate_idea(examples):
    """
    Generates a new, clickbaity YouTube documentary idea using GPT-4o.
    """
    example_str = ", ".join(examples)
    prompt = (
        f"Generate a new, clickbaity idea for a 3D YouTube documentary, similar to the style of the channel 'fern'. "
        f"The idea should be completely different from these examples: {example_str}. "
        f"It needs to be intriguing and make people want to click on it."
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a creative assistant that generates viral YouTube video ideas."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096
    }

    response = requests.post(API_URL, headers=headers, json=data)
    response_json = response.json()
    idea = response_json['choices'][0]['message']['content'].strip()
    return idea

def generate_narration(idea, style_guide_text):
    """
    Generates a 10-minute narration script containing only the narrator's speech.
    """
    words_for_10_min = 10 * 150
    prompt = (
        f"Write a narration script for a 10-minute YouTube documentary about '{idea}'. "
        f"The script should be approximately {words_for_10_min} words long. "
        f"**CRITICAL: Output ONLY the words the narrator speaks - nothing else.** "
        f"Do NOT include any of the following:\n"
        f"- Scene descriptions or stage directions in brackets like [Opening Scene]\n"
        f"- Speaker labels like 'Narrator:', 'Dr. Smith:', or any colons\n"
        f"- Interview segments or quotes from other people\n"
        f"- Music cues or sound descriptions\n"
        f"- Any text in brackets, parentheses, or asterisks\n"
        f"- Production notes or camera directions\n\n"
        f"Write ONLY what the narrator says aloud, as one continuous flowing narrative. "
        f"Study the example transcript carefully - notice how it's pure narration without any production elements. "
        f"Match that investigative, data-driven style that tells a compelling story with specific details and numbers. "
        f"Start with an immediate hook that grabs attention like the example does.\n\n"
        f"Example transcript style to follow:\n\n---\n{style_guide_text}\n---\n"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a scriptwriter for documentaries in the style of 'fern' YouTube channel. You write ONLY the pure narrator's speech - no brackets, no labels, no scene descriptions, no speaker tags. Output only the words spoken aloud by the narrator."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096
    }

    response = requests.post(API_URL, headers=headers, json=data)
    response_json = response.json()
    narration = response_json['choices'][0]['message']['content'].strip()
    return narration

def split_into_phrases(text):
    """
    Splits the narration script into a list of short phrases (3-5 seconds).
    """
    # An average speaking rate is about 150 words per minute (2.5 words/sec).
    # For a 3-5 second clip, we'll aim for phrases of about 8-15 words.
    words = text.replace('\n', ' ').split()
    phrases = []
    current_phrase = ""

    for word in words:
        if not current_phrase:
            current_phrase = word
        else:
            current_phrase += " " + word

        # If the phrase is in the target word count range and ends with a natural break, split it.
        if len(current_phrase.split()) >= 8 and word.endswith(('.', '!', '?', ',')):
            phrases.append(current_phrase)
            current_phrase = ""
        # Force a split if the phrase gets too long.
        elif len(current_phrase.split()) >= 15:
            phrases.append(current_phrase)
            current_phrase = ""

    # Add any remaining words as the last phrase.
    if current_phrase:
        phrases.append(current_phrase)
        
    return phrases

def _generate_single_prompt(phrase_with_index):
    """
    Helper function to generate a single prompt. Runs in a separate thread.
    """
    i, phrase = phrase_with_index
    prompt_for_flux = (
        f"Based on the following narration phrase, create a single, detailed visual prompt for a text-to-image model like FLUX. "
        f"The image will be used to generate a short 3-5 second video clip, so the scene should be concise and focused.\n\n"
        f"**Instructions for the prompt:**\n"
        f"1. **Visuals Only:** Describe only what can be seen. Do not include sound, camera movements, or non-visual elements.\n"
        f"2. **Single Scene:** The prompt must describe a single, static scene.\n"
        f"3. **Low-Poly Style:** The art style must be 'low-poly animation'.\n"
        f"4. **Object Detail:** Be very specific about the appearance and position of the main objects in the scene. Keep the background simpler.\n"
        f"5. **Concise:** The prompt should be a single paragraph.\n\n"
        f"**Narration Phrase:** \"{phrase}\"\n\n"
        f"Generate the prompt:"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an expert in writing prompts for AI image generation models. You create purely visual prompts for a low-poly animation style."},
            {"role": "user", "content": prompt_for_flux}
        ],
        "max_tokens": 4096
    }
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
        flux_prompt = response_json['choices'][0]['message']['content'].strip()
        end_time = time.time()
        request_time = end_time - start_time
        return (i, flux_prompt, request_time)
    except requests.exceptions.RequestException as e:
        print(f"Error for phrase {i+1}: {e}")
        return (i, None, 0)

def generate_flux_prompts(phrases):
    """
    Generates a single, visual prompt for each phrase concurrently.
    
    Returns:
        tuple[list[str], list[float]]: A tuple containing the list of prompts
                                       and a list of timings for each request.
    """
    prompts = []
    timings = []
    
    print(f"Generating {len(phrases)} prompts concurrently...")
    start_total_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Pass each phrase with its index to the worker function
        phrases_with_indices = list(enumerate(phrases))
        results = list(executor.map(_generate_single_prompt, phrases_with_indices))
        
    end_total_time = time.time()
    print(f"Finished generating all prompts in {end_total_time - start_total_time:.2f}s")
    
    # Process results, which are now ordered
    results.sort(key=lambda x: x[0])
    
    for i, flux_prompt, request_time in results:
        if flux_prompt:
            print(f"\n--- Prompt for phrase {i+1} (took {request_time:.2f}s) ---\n{flux_prompt}")
            prompts.append(flux_prompt)
            timings.append(request_time)
        else:
            # Append placeholders for failed requests to maintain list length
            prompts.append("Error generating prompt.")
            timings.append(0)

    return prompts, timings

def get_next_generation_number():
    """
    Gets the next generation number by checking existing folders in the outputs directory.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, 'outputs')
    
    if not os.path.exists(base_dir):
        return 1
    
    existing_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f[0].isdigit()]
    if not existing_folders:
        return 1
    
    # Extract generation numbers from folder names
    generation_numbers = []
    for folder in existing_folders:
        match = re.match(r'^(\d+)_', folder)
        if match:
            generation_numbers.append(int(match.group(1)))
    
    if not generation_numbers:
        return 1
    
    return max(generation_numbers) + 1

def create_safe_folder_name(idea, generation_number):
    """
    Creates a safe folder name from the idea title and generation number.
    """
    # Remove or replace unsafe characters for folder names
    safe_title = re.sub(r'[^\w\s-]', '', idea)  # Remove special chars except spaces and hyphens
    safe_title = re.sub(r'[-\s]+', '_', safe_title)  # Replace spaces and hyphens with underscores
    safe_title = safe_title.strip('_')  # Remove leading/trailing underscores
    
    # Limit length to avoid filesystem issues
    if len(safe_title) > 50:
        safe_title = safe_title[:50].rstrip('_')
    
    return f"{generation_number}_{safe_title}"

def save_to_json(idea, narration, phrases, prompts):
    """
    Saves all the generated content into a unique folder for this generation.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, 'outputs')
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Get the next generation number and create folder name
    generation_number = get_next_generation_number()
    folder_name = create_safe_folder_name(idea, generation_number)
    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCreated folder: '{output_dir}'")

    # Save the pure narration to a separate text file
    narration_file_path = os.path.join(output_dir, "narration.txt")
    with open(narration_file_path, "w", encoding='utf-8') as f:
        f.write(narration)
    
    print(f"Saved narration to '{narration_file_path}'.")

    # Save the idea to a separate text file
    idea_file_path = os.path.join(output_dir, "idea.txt")
    with open(idea_file_path, "w", encoding='utf-8') as f:
        f.write(idea)
    
    print(f"Saved idea to '{idea_file_path}'.")

    # Save the prompts to a separate text file
    prompts_file_path = os.path.join(output_dir, "prompts.txt")
    with open(prompts_file_path, "w", encoding='utf-8') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"--- Prompt {i+1} ---\n{prompt}\n\n")
    
    print(f"Saved prompts to '{prompts_file_path}'.")

    # Combine phrases and prompts into a list of scene objects
    scenes = []
    for phrase, prompt in zip(phrases, prompts):
        scenes.append({"phrase": phrase, "prompt": prompt})

    # Structure the final JSON output
    output_data = {
        "idea": idea,
        "narration_script": narration,
        "scenes": scenes,
        "generation_number": generation_number,
        "folder_name": folder_name
    }

    # Write the data to a JSON file
    json_file_path = os.path.join(output_dir, "documentary_data.json")
    with open(json_file_path, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved complete data to '{json_file_path}'.")
    
    return output_dir

if __name__ == "__main__":
    example_topics = [
        "The Lost City of Atlantis",
        "The Secrets of the Bermuda Triangle",
        "The Great Pyramid of Giza's Hidden Chambers"
    ]
    
    # 1. Generate the idea and narration
    print("Generating a new documentary idea...")
    new_idea = generate_idea(example_topics)
    print(f"Generated Idea: {new_idea}")
    
    # Construct an absolute path to the template file to ensure it's found
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    template_path = os.path.join(project_root, 'templates', 'fern_v1.txt')
    
    # Read the style guide text from the file
    with open(template_path, "r", encoding="utf-8") as f:
        style_guide = f.read()
        
    print("\nGenerating the narration script...")
    narration_script = generate_narration(new_idea, style_guide)
    
    # 2. Split the narration for prompt generation
    print("\nSplitting the narration into phrases...")
    narration_phrases = split_into_phrases(narration_script)
    
    # 3. Generate prompts and print them in real-time
    print("\nGenerating FLUX prompts for each phrase...")
    flux_prompts, _ = generate_flux_prompts(narration_phrases)
    
    # 4. Save all content to a unique generation folder
    output_folder = save_to_json(new_idea, narration_script, narration_phrases, flux_prompts)
    
    print(f"\nAll done! Check the '{output_folder}' folder for the results.")
