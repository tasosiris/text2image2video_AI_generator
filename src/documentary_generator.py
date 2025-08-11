import os
import requests
import json
import re
from dotenv import load_dotenv
import time
import concurrent.futures
from functools import partial
import sys
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

# Get the API key and base URL for AIMLAPI (OpenAI-compatible)
API_KEY = os.getenv("AIML_API_KEY")
AIML_BASE_URL = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
MODEL_ID = os.getenv("AIML_MODEL_ID", "gpt-4o")

if not API_KEY:
    print("Warning: AIML_API_KEY is not set. Requests will fail with 401/403.", file=sys.stderr)

# OpenAI-compatible client pointing to AIMLAPI per docs:
# https://docs.aimlapi.com/api-references/text-models-llm/openai
aiml_client = OpenAI(api_key=API_KEY, base_url=AIML_BASE_URL)


def generate_idea(examples, visual_style: str = "photorealistic"):
    """
    Generates a new documentary idea, theme, and visual style guide using GPT-4o.
    The visual style guide adapts to the requested visual_style ("photorealistic" or "low_poly").
    
    Returns:
        tuple[str, str, str]: A tuple containing the idea, theme, and visual style guide.
    """
    example_str = ", ".join(examples)
    style_key = (visual_style or "").strip().lower()

    if style_key in {"low_poly", "low-poly", "low poly", "lowpoly"}:
        aesthetic_line = (
            "Aesthetic: Strictly 'low‑poly 3D animation style'. Stylized with clean lines, simple geometric shapes, and flat shading. Absolutely no photorealism."
        )
        character_line = (
            "Character Styling: All humans MUST be depicted in a consistent low‑poly style with simplified, geometric, often faceless features. Avoid realistic, cartoonish, or blocky/lego‑like looks."
        )
        env_line = (
            "Environment & Objects: Minimalist, composed of simple geometric shapes. Avoid clutter; keep strong silhouettes and simple forms."
        )
        palette_line = (
            "Color Palette: A limited palette (e.g., earthy terracottas, marble whites/greys, deep blues/greens) with soft, ambient lighting."
        )
    else:
        # photorealistic default
        aesthetic_line = (
            "Aesthetic: Photorealistic documentary look with natural lighting, physically plausible materials, and true‑to‑life textures. Absolutely no low‑poly, cartoon, or stylized looks."
        )
        character_line = (
            "Character Styling: Realistic, human‑like faces and skin tones with natural proportions. Avoid toy‑like, blocky, anime, voxel, or stylized traits."
        )
        env_line = (
            "Environment & Objects: Real‑world materials and details; avoid simplified geometric abstractions. Composition favors clarity and realism over stylization."
        )
        palette_line = (
            "Color Palette: Cinematic yet natural palette appropriate to the scene (e.g., soft daylight, warm interiors, cool evening tones) with plausible shadows and highlights."
        )

    prompt = (
        f"Generate a new, clickbaity idea for a YouTube documentary, similar to the narrative style of the channel 'fern'. "
        f"CRITICAL DOMAIN CONSTRAINT: The topic must be strictly about Ancient Greek–Roman history and adjacent Mediterranean cultures circa 800 BCE–500 CE "
        f"(e.g., Classical and Hellenistic Greece, the Roman Republic and Empire, the Etruscans, Carthage, and interactions with Persia). "
        f"Do NOT include modern topics or cultures outside this region and timeframe. "
        f"The idea should be completely different from these examples: {example_str}. "
        f"Please return the output as a JSON object with three keys: 'idea', 'theme', and 'visual_style_guide'.\n\n"
        f"1.  **idea**: A single, extremely clickbaity title for the documentary, ideally 5-10 words long.\n"
        f"2.  **theme**: 1-3 words describing the mood (e.g., 'Mysterious, suspenseful', 'Dark, academic', 'Ambient, awe-inspiring').\n"
        f"3.  **visual_style_guide**: A detailed paragraph for an art director. This guide is CRITICAL for maintaining consistency. It must define:\n"
        f"    -   **{aesthetic_line}**\n"
        f"    -   **{character_line}**\n"
        f"    -   **{env_line}**\n"
        f"    -   **{palette_line}**\n\n"
        f"Your response MUST be a valid JSON object."
    )

    try:
        response = aiml_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a creative assistant that generates viral YouTube video ideas and provides art direction for a low-poly 3D animation style. You must output a valid JSON object."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
    except Exception as e:
        print(f"AIMLAPI request failed: {e}", file=sys.stderr)
        raise

    response_json = response.model_dump() if hasattr(response, "model_dump") else response
    
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
    Generates a 5-6 minute narration script containing only the narrator's speech.
    """
    words_for_5_to_6_min = int(5.5 * 150) # Average 5.5 minutes at 150 wpm
    prompt = (
        f"Write a narration script for a 5-to-6-minute YouTube documentary about '{idea}'. "
        f"The script should be approximately {words_for_5_to_6_min} words long. "
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
        f"CRITICAL: Write ONLY what the narrator says aloud. Every sentence MUST be between 10 and 15 words long. "
        f"Allow commas and occasional em-dashes for flow, but keep clarity high and sentence length consistent. "
        f"Avoid choppy, fragmentary lines; let thoughts complete within sentences. "
        f"Begin with a brief YouTube-style intro hook (1–2 sentences) that frames the topic and stakes for the viewer. "
        f"Study the example transcript carefully - notice how it's pure narration without any production elements. "
        f"Match that investigative, data-driven style that tells a compelling story with specific details and numbers. "
        f"Conclude with a natural YouTube call-to-action in one short sentence (e.g., invite the viewer to like, subscribe, and comment), keeping tone consistent and not salesy. "
        f"Avoid saying channel names or using brackets.\n\n"
        f"Example transcript style to follow:\n\n---\n{style_guide_text}\n---\n"
    )

    try:
        response = aiml_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a scriptwriter for documentaries in the style of 'fern' YouTube channel. You write ONLY the pure narrator's speech - no brackets, no labels, no scene descriptions, no speaker tags. Keep strictly to Ancient Greek–Roman and adjacent Mediterranean history (c. 800 BCE–500 CE). Output only the words spoken aloud by the narrator."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )
    except Exception as e:
        print(f"AIMLAPI request failed: {e}", file=sys.stderr)
        raise
    response_json = response.model_dump() if hasattr(response, "model_dump") else response
    narration = response_json['choices'][0]['message']['content'].strip()
    return narration

def split_into_phrases(text: str) -> list[str]:
    """
    Split narration into natural phrases:
    - Primary split on sentence enders (. ! ?)
    - Merge very short trailing sentences (<= 3 words or <= 20 chars) with the previous sentence
      so fragments like "in the ancient world." attach to the prior phrase.
    - Do NOT enforce duration limits (removes 3–4 sec caps).
    - Avoid splitting on commas; keep author-intended sentence rhythm.
    """
    if not text or not text.strip():
        return []

    single_line = re.sub(r"\s+", " ", text.strip())
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", single_line) if s.strip()]

    def is_tiny_fragment(s: str) -> bool:
        words = re.findall(r"\b\w+[\w'-]*\b", s)
        return len(words) <= 3 or len(s) <= 20

    merged: list[str] = []
    for s in sentences:
        if merged and is_tiny_fragment(s):
            merged[-1] = f"{merged[-1]} {s}".strip()
        else:
            merged.append(s)

    # Final tidy-up: collapse stray spaces
    return [m.strip() for m in merged if m.strip()]

def _generate_single_prompt(phrase_with_index, theme, visual_style_guide, idea, visual_style: str = "photorealistic"):
    """
    Helper function to generate a single prompt that adheres to a theme and style guide.
    """
    i, phrase = phrase_with_index
    style_key = (visual_style or "").strip().lower()

    # Define style-specific system messages and prompt instructions
    if style_key in {"low_poly", "low-poly", "lowpoly", "low poly"}:
        system_message_content = (
            "You are an expert in writing prompts for AI image and video generation models. "
            "Your task is to create purely visual prompts for a low-poly 3D animation style. "
            "You must translate a given narration sentence into a compelling, static visual scene. "
            "Your primary goal is to ensure visual consistency, simplicity, and direct contextual relevance to the narration."
        )
        prompt_for_flux = (
            f"Your task is to create a single, concise visual prompt for a cinematic scene. This scene must be a direct and compelling visualization of the following narration sentence: \"{phrase}\". "
            f"The scene is for a low-poly 3D documentary titled '{idea}' with a '{theme}' tone. "
            f"Use plain, concrete language. Describe only what is visible; avoid metaphors, poetic wording, and implied sounds. "
            f"Do not use the words 'shot', 'frame', or 'series'. "
            f"Include one cinematic composition term (e.g., wide shot, extreme close-up, aerial). Specify the main subject, clear colors/materials, and a minimal background. "
            f"Match mood and lighting to the phrase. Indicate camera‑only movement (parallax feel) and keep all subjects/objects static. "
            f"If humans are present, describe simple clothing colors and forms; no silhouettes or blacked‑out figures. "
            f"Strictly follow this visual style guide:\n{visual_style_guide}\n"
            f"Constraints: low‑poly only; at most three people in frame; avoid crowds; static human figures; avoid complex textures; avoid photorealism; no moving subjects or objects; no silhouettes. "
            f"Output ONLY the final prompt text, without labels or markdown."
        )
    else:  # photorealistic default
        system_message_content = (
            "You are an expert in writing prompts for AI image and video generation models. "
            "Your task is to create purely visual prompts for a photorealistic documentary. "
            "You must translate a given narration sentence into a compelling, cinematic, and realistic visual scene with static subjects. "
            "Use simple, direct language. Avoid metaphors. Focus on clear, concrete descriptions."
        )
        prompt_for_flux = (
            f"CRITICAL: Start with a 'slow parallax shot'. This is mandatory. "
            f"Your task is to create a single, concise visual prompt for a cinematic scene. This scene must be a direct and compelling visualization of the following narration sentence: \"{phrase}\". "
            f"The scene is for a photorealistic documentary titled '{idea}' with a '{theme}' tone. "
            f"Use plain, concrete language. Describe only what is visible; avoid metaphors, poetic wording, and implied sounds. "
            f"Do not use the words 'shot', 'frame', or 'series'. "
            f"Include one cinematic composition term (e.g., wide shot, extreme close-up, aerial). Specify the main subject, clear colors/materials, and a minimal background. "
            f"Match mood and lighting to the phrase. Emphasize cinematic, realistic lighting (e.g., 'golden hour', 'Rembrandt lighting') and specify realistic surface textures (e.g., 'rough-hewn stone', 'polished marble'). "
            f"There should be no fast camera movements, and any people in the scene should have static, non-moving hands. "
            f"Faces must be in high definition and as realistic as possible. "
            f"When describing people, use simple, direct terms. For women, describe their key characteristics, such as facial features, posture, and clothing, instead of just the environment. "
            f"If people are present, specify skin tone and clothing colors; no silhouettes or blacked-out figures. "
            f"Constraints: photorealistic, true human‑like faces and skin; natural materials and lighting; at most three people in frame; avoid crowds; avoid cartoon, anime, toy, low‑poly, blocky, lego, voxel, stylized, or illustrated looks; no moving subjects or objects; no silhouettes. "
            f"Output ONLY the final prompt text, without labels or markdown."
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = aiml_client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": prompt_for_flux},
                ],
                max_tokens=4096,
            )
            response_json = response.model_dump() if hasattr(response, "model_dump") else response
            flux_prompt = response_json['choices'][0]['message']['content'].strip()
            # Clean up any residual markdown or prefixes from the model's output
            flux_prompt = re.sub(r'^\*+.*?\*+[:\s]*', '', flux_prompt)
            end_time = time.time()
            request_time = end_time - start_time
            return (i, flux_prompt, request_time)
        except Exception as e:
            print(f"Error for phrase {i+1} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Exponential backoff
            else:
                return (i, "Error generating prompt.", 0)
    
    return (i, "Error generating prompt.", 0)

def generate_flux_prompts(phrases, theme, visual_style_guide, idea=None, visual_style: str = "photorealistic"):
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
    
    # Use functools.partial to pass theme, style guide, idea, and visual_style to the worker function
    worker_func = partial(
        _generate_single_prompt,
        theme=theme,
        visual_style_guide=visual_style_guide,
        idea=idea,
        visual_style=visual_style,
    )
    
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
