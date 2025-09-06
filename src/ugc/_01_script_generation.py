import os
import json
import sys
import re
import random
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Tuple, Dict, Any

# Load environment variables from a .env file.
# This allows you to store sensitive information like API keys securely.
load_dotenv()

# Get the API key and base URL for AIMLAPI from environment variables.
# Using os.getenv allows for default values if the variables are not set.
API_KEY = os.getenv("AIML_API_KEY")
AIML_BASE_URL = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
MODEL_ID = os.getenv("AIML_MODEL_ID", "gpt-5")

# A check to ensure the API key is set, warning the user if it's not.
if not API_KEY:
    print("Warning: AIML_API_KEY is not set. Requests will likely fail.", file=sys.stderr)

# Initialize the OpenAI client, pointing it to the AIMLAPI endpoint.
# This makes the code compatible with OpenAI's library while using AIMLAPI's service.
aiml_client = OpenAI(api_key=API_KEY, base_url=AIML_BASE_URL)


def generate_ugc_script(product_details: dict, creator_profile: dict):
    """
    Generates a UGC video script using AIMLAPI based on product and creator details.

    Args:
        product_details (dict): A dictionary containing information about the product.
        creator_profile (dict): A dictionary containing the profile of the UGC creator.

    Returns:
        str: The generated video script as a string.
    """
    # Combines the creator's personality traits into a readable string.
    personality_str = ", ".join(creator_profile['style_personality'].values())

    # The main prompt sent to the language model. It is constructed using f-strings
    # to insert the product and creator details directly into the text.
    prompt = f"""
You are a TikTok UGC video scriptwriter.
Your goal is to create a 30-35 second script that sells a product in an authentic, user-generated content (UGC) style.

### Creator Persona:
- **Name:** {creator_profile['name']}
- **Style:** {creator_profile['tagline']}
- **Personality:** {personality_str}

### Requirements:
- The script must be structured as scenes with timestamps.
- **Hook:** Start with a strong, scroll-stopping hook that feels like dropping into the middle of a thought. For example, instead of a polished question like 'Ever opened a bag of chips to find them stale?', try something more direct and personal like 'Okay, I have to show you this,' or 'I'm never buying stale chips again because of this.' Avoid generic introductions.
- **Pacing:** The total narration should be between 100 and 140 words to fit the 30-35 second video length.
- **Content:** Each scene must specify four things: narration, scene type, an AI Video Prompt, and a Text Overlay.
- **Style:** Narration must be extremely casual and sound like a real person talking to a friend, not a polished ad script. Use conversational language and a relatable tone. Each scene's narration should be a complete thought of about 12-18 words. The narration text must not be wrapped in quotation marks. To ensure correct pronunciation by text-to-speech, expand numbers and symbols into full words (e.g., write 'three times' instead of '3x'). Do not use the phrase 'dorm life'. Product shots are always shown separately from the person talking.
- **Formatting:** Absolutely no emojis.
- **CTA:** End with a natural, clear call to action suitable for a TikTok Shop affiliate link (e.g., 'Check the link below,' 'Tap the orange cart to shop'). Avoid 'link in bio'.
- **AI Video Prompt:** This is a prompt for an AI video generation model. For 'UGC talking' scenes, the prompt must ONLY describe the person and their emotion. Do not include any background, setting, or other objects. Always describe the creator as looking excited or enthusiastic. For example: 'A friendly, relatable young white woman in her mid-20s, speaking excitedly to the camera. Casual look, authentic user-generated content style.' For 'Product shot' scenes, the prompt must be cinematic and descriptive, including composition, subject, action, and lighting.
- **Text Overlay:** Use the `Text overlay` field for any on-screen text. If a scene has no text, leave it empty.

### Input:
Product: {product_details['name']}
Niche: {product_details['niche']}
Target audience: {product_details['audience']}
Tone: {product_details['tone']}
Main problem solved: {product_details['problem']}
Key features/benefits: {', '.join(product_details['features'])}
Call to action: {product_details['cta']}

### Output Format:
Return as a structured list of scenes following this exact format:

Scene {{{{number}}}}
- **Timestamp (estimated sec):** {{{{start–end}}}}
- **Narration:** {{{{spoken words}}}}
- **Scene type:** {{{{UGC talking | Product shot}}}}
- **AI Video Prompt:** {{{{prompt for AI video generation}}}}
- **Text overlay:** {{{{text to display on screen, or empty}}}}
"""

    try:
        # Makes the actual API call to the AIMLAPI service.
        response = aiml_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a TikTok UGC video scriptwriter. Your goal is to create a short 15–30 second script that sells a product in an authentic, user-generated content (UGC) style."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,  # Limits the length of the response.
        )
        # Extracts the generated script text from the API response.
        script = response.choices[0].message.content.strip()
        return script
    except Exception as e:
        # Handles potential errors during the API request, like network issues or auth failures.
        print(f"AIMLAPI request failed: {e}", file=sys.stderr)
        return "Error: Could not generate script."

def parse_script_to_structured_data(script_text: str) -> list[dict]:
    """
    Parses the raw text script from the AI into a list of scene dictionaries.
    This makes the data structured and machine-readable for other tools.
    """
    scenes = []
    # Split the script by "Scene " followed by a number to get individual scene blocks.
    # This is more robust than splitting by just a newline.
    scene_blocks = re.split(r'Scene \d+\s*\n', script_text)
    
    for block in scene_blocks:
        # Ignore any empty blocks that result from the split.
        if not block.strip():
            continue
            
        scene_data = {}
        # Use regular expressions to find and extract each piece of data.
        # The `re.DOTALL` flag allows `.` to match newlines, making the regex more robust.
        # `re.search` finds the first match for each pattern in the block.
        timestamp_match = re.search(r'- \*\*Timestamp \(estimated sec\):\*\*\s*(.*?)\n', block, re.DOTALL)
        narration_match = re.search(r'- \*\*Narration:\*\*\s*(.*?)\n', block, re.DOTALL)
        scene_type_match = re.search(r'- \*\*Scene type:\*\*\s*(.*?)\n', block, re.DOTALL)
        ai_prompt_match = re.search(r'- \*\*AI Video Prompt:\*\*\s*(.*?)\n', block, re.DOTALL)
        # For the last item, we don't need a newline at the end.
        text_overlay_match = re.search(r'- \*\*Text overlay:\*\*\s*(.*)', block, re.DOTALL)
        
        # Extract the matched group and clean up whitespace.
        if timestamp_match:
            scene_data['timestamp'] = timestamp_match.group(1).strip()
        if narration_match:
            scene_data['narration'] = narration_match.group(1).strip()
        if scene_type_match:
            scene_data['scene_type'] = scene_type_match.group(1).strip()
        if ai_prompt_match:
            scene_data['ai_video_prompt'] = ai_prompt_match.group(1).strip()
        if text_overlay_match:
            scene_data['text_overlay'] = text_overlay_match.group(1).strip()
        
        # Only add the scene if we successfully extracted data.
        if scene_data:
            scenes.append(scene_data)
            
    return scenes

def create_safe_name(name: str) -> str:
    """
    Creates a safe string for file or folder names by removing special
    characters and replacing spaces with underscores.
    """
    # Removes characters that are not alphanumeric, spaces, or hyphens.
    safe_name = re.sub(r'[^\w\s-]', '', name)
    # Replaces one or more spaces or hyphens with a single underscore.
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    # Truncate the name to a reasonable length to avoid filesystem errors
    safe_name = safe_name[:75]
    # Removes any leading or trailing underscores that might have been created.
    return safe_name.strip('_')

def save_script_to_json(creator_name: str, product_name: str, product_id: str, script_data: list[dict]) -> Optional[str]:
    """
    Saves the structured script data to a JSON file.
    The structure will be: outputs/ugc_scripts/CREATOR_NAME/YYYY-MM-DD_PRODUCT_ID/PRODUCT_NAME_script.json
    Returns:
        The path to the saved JSON file, or None on failure.
    """
    try:
        # Determine the project's root directory to correctly place the 'outputs' folder.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))

        # Create safe, valid names for the folder and file.
        safe_creator_name = create_safe_name(creator_name)
        safe_product_name = create_safe_name(product_name)

        # Get the current date to include in the folder name.
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Define the name for the new script-specific folder.
        script_folder_name = f"{date_str}_{product_id}"
        
        # Construct the full path for the new output directory.
        output_dir = os.path.join(project_root, 'outputs', 'ugc_scripts', safe_creator_name, script_folder_name)
        
        # Create all necessary directories in the path if they don't already exist.
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the full path for the output file.
        file_path = os.path.join(output_dir, f"{safe_product_name}_script.json")
        
        # Write the structured data to the JSON file.
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(script_data, f, indent=4, ensure_ascii=False)
        
        print(f"\nScript successfully saved to: {file_path}")
        return file_path

    except Exception as e:
        print(f"Error: Failed to save the script to a file. {e}", file=sys.stderr)
        return None


def select_product_and_creator() -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Selects a product with the minimum 'videos_generated' count and its matched creator.

    Returns:
        A tuple containing the selected product and creator profile, or (None, None) on failure.
    """
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        matched_products_path = os.path.join(project_root, 'outputs', 'products', 'products_data.json')

        with open(matched_products_path, 'r', encoding='utf-8') as f:
            products = json.load(f)

        # Filter for products with a matched profile and sort by videos_generated
        eligible_products = [p for p in products if p.get('matched_profile')]
        if not eligible_products:
            print("No products with matched profiles found.", file=sys.stderr)
            return None, None

        sorted_products = sorted(eligible_products, key=lambda p: p.get('videos_generated', 0))
        selected_product = sorted_products[0]

        # Load creator profiles and find the matched one
        profiles_path = os.path.join(script_dir, 'profiles.json')
        with open(profiles_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)

        creator_name = selected_product['matched_profile']
        creator_profile = next((p for p in profiles if p['name'] == creator_name), None)

        if not creator_profile:
            print(f"Creator profile for '{creator_name}' not found.", file=sys.stderr)
            return selected_product, None

        return selected_product, creator_profile

    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}", file=sys.stderr)
        return None, None
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error reading or parsing JSON file: {e}", file=sys.stderr)
        return None, None


def format_product_for_prompt(product: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats the selected product data into the structure expected by the script generation prompt.
    """
    product_text = product.get("Products", "")
    
    # Extract a clean product name
    if "Price:" in product_text:
        product_name = product_text.split("Price:")[0].strip()
    else:
        product_name = product_text.split(',')[0].strip()
    
    product_name = re.sub(r'\[.*?\]', '', product_name).strip()

    # Generic details based on category (can be expanded)
    category = product.get("category", "General")
    niche = f"E-commerce / {category}"
    audience = "General audience on TikTok"
    tone = "Authentic, engaging, and relatable"
    problem = f"Finding a great product in the {category.lower()} space."
    features = [f"High-quality {category.lower()} product", "Great value", "Popular online"]
    cta = "Urge the user to buy the product from the TikTok Shop link."

    return {
        "name": product_name,
        "niche": niche,
        "audience": audience,
        "tone": tone,
        "problem": problem,
        "features": features,
        "cta": cta,
        "product_id": product.get("product_id")
    }


def update_videos_generated_count(product_id: str):
    """
    Increments the 'videos_generated' count for the specified product.
    """
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        matched_products_path = os.path.join(project_root, 'outputs', 'products', 'products_data.json')

        with open(matched_products_path, 'r', encoding='utf-8') as f:
            products = json.load(f)

        product_found = False
        for product in products:
            if product.get('product_id') == product_id:
                product['videos_generated'] = product.get('videos_generated', 0) + 1
                product_found = True
                break
        
        if product_found:
            with open(matched_products_path, 'w', encoding='utf-8') as f:
                json.dump(products, f, indent=4)
            print(f"Incremented 'videos_generated' count for product_id: {product_id}")
        else:
            print(f"Warning: Product with id '{product_id}' not found for updating count.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: '{matched_products_path}' not found.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while updating video count: {e}", file=sys.stderr)


def run_generate_ugc_script() -> Optional[tuple[str, str]]:
    """
    Runs the full UGC script generation process.

    Returns:
        A tuple containing the path to the generated JSON file and the product image filename, or None on failure.
    """
    print("--- Selecting Product and Creator for Script Generation ---")
    selected_product, creator_profile_to_use = select_product_and_creator()

    if not selected_product or not creator_profile_to_use:
        print("Could not select a product or creator. Aborting script generation.", file=sys.stderr)
        return None

    product_details = format_product_for_prompt(selected_product)
    product_id = product_details.get("product_id")
    product_image_filename = selected_product.get("product_image_filename")

    if not product_id or not product_image_filename:
        print("Error: Selected product is missing 'product_id' or 'product_image_filename'. Aborting.", file=sys.stderr)
        return None

    try:
        print("Generating UGC script with the following details:")
        print(f"Product: {product_details['name']}")
        print(f"Creator: {creator_profile_to_use['name']}")
        print("-" * 30)

        # Call the generation function with the product data.
        generated_script_text = generate_ugc_script(product_details, creator_profile_to_use)

        # Print the final raw script to the console for visibility.
        print("\n--- Generated UGC Script (Raw Text) ---")
        print(generated_script_text)

        # If the script was generated successfully, parse and save it to a JSON file.
        if "Error:" not in generated_script_text:
            # Step 1: Parse the raw text into structured data (a list of dictionaries).
            structured_script_data = parse_script_to_structured_data(generated_script_text)
            
            # Step 2: If parsing was successful, save the structured data.
            if structured_script_data:
                json_path = save_script_to_json(
                    creator_name=creator_profile_to_use['name'],
                    product_name=product_details['name'],
                    product_id=product_id,
                    script_data=structured_script_data
                )
                
                # Step 3: Increment the video count for the product
                if json_path:
                    update_videos_generated_count(product_id)
                
                return json_path, product_image_filename
            else:
                # Add a check in case parsing fails to produce any data.
                print("Error: Failed to parse the generated script text into a structured format.", file=sys.stderr)
                return None
        else:
            return None

    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}", file=sys.stderr)
        return None

# This block runs only when the script is executed directly (not when imported).
if __name__ == "__main__":
    run_generate_ugc_script()
