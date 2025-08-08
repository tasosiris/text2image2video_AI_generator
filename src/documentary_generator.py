import os
import requests
import json
import re
from dotenv import load_dotenv
import time
import concurrent.futures
from functools import partial
import sys

# Load environment variables from a .env file
load_dotenv()

# Get the API key from environment variables
API_KEY = os.getenv("AIML_API_KEY")
API_URL = "https://api.aimlapi.com/v1/chat/completions"


def generate_idea(examples):
    """
    Generates a new documentary idea, theme, and visual style guide using GPT-4o.
    
    Returns:
        tuple[str, str, str]: A tuple containing the idea, theme, and visual style guide.
    """
    example_str = ", ".join(examples)
    prompt = (
        f"Generate a new, clickbaity idea for a 3D YouTube documentary, similar to the style of the channel 'fern'. "
        f"CRITICAL DOMAIN CONSTRAINT: The topic must be strictly about Ancient Greek–Roman history and adjacent Mediterranean cultures circa 800 BCE–500 CE "
        f"(e.g., Classical and Hellenistic Greece, the Roman Republic and Empire, the Etruscans, Carthage, and interactions with Persia). "
        f"Do NOT include modern topics or cultures outside this region and timeframe. "
        f"The idea should be completely different from these examples: {example_str}. "
        f"Please return the output as a JSON object with three keys: 'idea', 'theme', and 'visual_style_guide'.\n\n"
        f"1.  **idea**: A single, catchy title for the documentary.\n"
        f"2.  **theme**: 1-3 words describing the mood (e.g., 'Mysterious, suspenseful', 'Dark, academic', 'Ambient, awe-inspiring').\n"
        f"3.  **visual_style_guide**: A detailed paragraph for an art director. This guide is CRITICAL for maintaining consistency. It must define:\n"
        f"    -   **Aesthetic**: Strictly 'low-poly 3D animation style'. Describe it as 'Stylized with clean lines, simple geometric shapes, and flat shading. Absolutely no photorealism.'\n"
        f"    -   **Character Styling**: This is the most important rule. All humans MUST be depicted in a consistent low-poly style. They should have simplified, geometric, and often faceless features. CRITICAL: Avoid realistic, cartoonish, or blocky/lego-like looks. Every character in every scene must look like they belong to the same set of low-poly models.\n"
        f"    -   **Environment & Objects**: Environments and objects must be minimalist, composed of simple geometric shapes. Scenes must not be cluttered to maintain focus.\n"
        f"    -   **Color Palette**: A specific, limited color palette (e.g., 'Earthy tones with deep blues and greens') with soft, ambient lighting to create a consistent mood.\n\n"
        f"Your response MUST be a valid JSON object."
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a creative assistant that generates viral YouTube video ideas and provides art direction for a low-poly 3D animation style. You must output a valid JSON object."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096
    }

    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    response_json = response.json()
    
    try:
        content_str = response_json['choices'][0]['message']['content']
        content = json.loads(content_str)
        idea = content['idea']
        theme = content['theme']
        style_guide = content['visual_style_guide']
        return idea, theme, style_guide
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON response from API: {e}", file=sys.stderr)
        print(f"Raw response content: {response_json['choices'][0]['message']['content']}", file=sys.stderr)
        raise

def generate_narration(idea, style_guide_text):
    """
    Generates a 3-minute narration script containing only the narrator's speech.
    """
    words_for_3_min = 3 * 150
    prompt = (
        f"Write a narration script for a 3-minute YouTube documentary about '{idea}'. "
        f"The script should be approximately {words_for_3_min} words long. "
        f"**CRITICAL: Output ONLY the words the narrator speaks - nothing else.** "
        f"CRITICAL DOMAIN CONSTRAINT: Keep the narrative strictly within Ancient Greek–Roman history and adjacent Mediterranean cultures circa 800 BCE–500 CE. "
        f"Avoid modern references, technologies, or events outside this region and timeframe. "
        f"Favor concrete historical details (dates, places, people, sources) when relevant. "
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
            {"role": "system", "content": "You are a scriptwriter for documentaries in the style of 'fern' YouTube channel. You write ONLY the pure narrator's speech - no brackets, no labels, no scene descriptions, no speaker tags. Keep strictly to Ancient Greek–Roman and adjacent Mediterranean history (c. 800 BCE–500 CE). Output only the words spoken aloud by the narrator."},
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
    It prioritizes splitting at sentence endings, then at commas in longer phrases.
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

        # Prioritize splitting at sentence endings ('.', '!', '?').
        if word.endswith(('.', '!', '?')):
            phrases.append(current_phrase)
            current_phrase = ""
        # If the phrase is getting long, split at a comma.
        elif len(current_phrase.split()) >= 8 and word.endswith(','):
            phrases.append(current_phrase)
            current_phrase = ""
        # Force a split if the phrase gets too long without any natural break.
        elif len(current_phrase.split()) >= 15:
            phrases.append(current_phrase)
            current_phrase = ""

    # Add any remaining words as the last phrase.
    if current_phrase:
        phrases.append(current_phrase)
        
    return phrases

def _generate_single_prompt(phrase_with_index, theme, visual_style_guide):
    """
    Helper function to generate a single prompt that adheres to a theme and style guide.
    """
    i, phrase = phrase_with_index
    prompt_for_flux = (
        f"You are a creative director for a high-quality 3D animated YouTube documentary. "
        f"Your task is to create a single, detailed visual prompt for a text-to-image model (FLUX) based on the following narration phrase. "
        f"The resulting image should feel like a still frame from a cinematic animation.\n\n"
        f"**Narration Phrase:** \"{phrase}\"\n\n"
        f"**Instructions for the visual prompt:**\n"
        f"1.  **Cinematic Composition:** Describe the scene from a specific camera perspective. Use terms like 'wide shot', 'extreme close-up', 'worm's-eye view', or 'aerial shot'. Emphasize a strong focal point and use principles like the rule of thirds to create a visually compelling composition.\n"
        f"2.  **Mood and Lighting:** The lighting must match the emotional tone of the narration. Use descriptive terms like 'dramatic and shadowy', 'soft and ethereal', 'warm and nostalgic', or 'cold and clinical'. Describe how light and shadow interact with the objects.\n"
        f"3.  **Visual Style Guide Adherence:** You MUST strictly follow this guide:\n{visual_style_guide}\n"
        f"4.  **Core Style:** The art style is 'low-poly animation'. This means clean, geometric shapes and sharp edges. It is crucial to AVOID photorealism, high-frequency textures, and excessive detail. The final image must not contain characters that look like 'Lego' figures, toys, or have a 'blocky' or 'Minecraft' appearance.\n"
        f"5.  **Creative Interpretation:** Your primary goal is to avoid visual repetition. Do not default to simple scenes of a figure looking at a planet. Think metaphorically and abstractly. How can you visually represent the *concept* in the narration? Consider creating scenes with symbolic machinery, abstract data visualizations, impossible architecture, or metaphorical landscapes that relate to the narration. Surprise the viewer with a creative, non-obvious interpretation.\n"
        f"6.  **Focused Detail:** Be very specific about the appearance and position of the main subject. The background should be simpler and support the main subject without distracting from it.\n"
        f"7.  **Human Count Constraint:** If humans are present, include at most three human-like figures in total. Avoid crowds, groups larger than three, duplicated figures, or background silhouettes that increase the count. Prefer a single subject or a small group of up to three.\n"
        f"8.  **Output Format:** The prompt must be a single, relatively simple paragraph. It must begin with the documentary's theme, followed by a description of what is visible in the frame. Your entire response must ONLY be this prompt text.\n\n"
        f"Theme: \"{theme}\"\n"
        f"CRITICAL: Generate the prompt now. Do not include any introductory text, labels, or markdown like '**Visual Prompt:**'."
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an expert in writing prompts for AI image and video generation models. You create purely visual prompts for a low-poly 3D animation style. You MUST include instructions for smooth camera movement and ensure any human figures are described as static (no movement). Your primary goal is to ensure visual consistency, simplicity, and contextual accuracy."},
            {"role": "user", "content": prompt_for_flux}
        ],
        "max_tokens": 4096
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=60)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            response_json = response.json()
            flux_prompt = response_json['choices'][0]['message']['content'].strip()
            # Clean up any residual markdown or prefixes from the model's output
            flux_prompt = re.sub(r'^\*+.*?\*+[:\s]*', '', flux_prompt)
            end_time = time.time()
            request_time = end_time - start_time
            return (i, flux_prompt, request_time)
        except requests.exceptions.RequestException as e:
            print(f"Error for phrase {i+1} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Exponential backoff
            else:
                return (i, "Error generating prompt.", 0)
    
    return (i, "Error generating prompt.", 0)

def generate_flux_prompts(phrases, theme, visual_style_guide):
    """
    Generates a single, visual prompt for each phrase concurrently, adhering to a theme.
    
    Returns:
        tuple[list[str], list[float]]: A tuple containing the list of prompts
                                       and a list of timings for each request.
    """
    prompts = []
    timings = []
    
    print(f"Generating {len(phrases)} prompts concurrently with theme '{theme}'...")
    start_total_time = time.time()
    
    # Use functools.partial to pass the theme and style guide to the worker function
    worker_func = partial(_generate_single_prompt, theme=theme, visual_style_guide=visual_style_guide)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        phrases_with_indices = list(enumerate(phrases))
        results = list(executor.map(worker_func, phrases_with_indices))
        
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

def save_to_json(idea, theme, visual_style_guide, narration, phrases, prompts):
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

    # Save the idea, theme, and style guide
    idea_file_path = os.path.join(output_dir, "idea_and_style.txt")
    with open(idea_file_path, "w", encoding='utf-8') as f:
        f.write(f"Idea: {idea}\n\n")
        f.write(f"Theme: {theme}\n\n")
        f.write(f"Visual Style Guide:\n{visual_style_guide}\n")
    print(f"Saved idea and style guide to '{idea_file_path}'.")

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
        "theme": theme,
        "visual_style_guide": visual_style_guide,
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
    
    # 1. Generate the idea, theme, and style guide
    print("Generating a new documentary idea...")
    new_idea, theme, style_guide = generate_idea(example_topics)
    print(f"Generated Idea: {new_idea}")
    print(f"Theme: {theme}")
    print(f"Visual Style Guide: {style_guide}")
    
    # Construct an absolute path to the template file to ensure it's found
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    template_path = os.path.join(project_root, 'templates', 'fern_v1.txt')
    
    # Read the style guide text from the file
    with open(template_path, "r", encoding="utf-8") as f:
        narration_style_guide = f.read()
        
    print("\nGenerating the narration script...")
    narration_script = generate_narration(new_idea, narration_style_guide)
    
    # 2. Split the narration for prompt generation
    print("\nSplitting the narration into phrases...")
    narration_phrases = split_into_phrases(narration_script)
    
    # 3. Generate prompts based on the theme and style guide
    print("\nGenerating FLUX prompts for each phrase...")
    flux_prompts, _ = generate_flux_prompts(narration_phrases, theme, style_guide)
    
    # 4. Save all content to a unique generation folder
    output_folder = save_to_json(new_idea, theme, style_guide, narration_script, narration_phrases, flux_prompts)
    
    print(f"\nAll done! Check the '{output_folder}' folder for the results.")
