"""
This script initiates the complete product pipeline by running the FastMoss login, data extraction,
and product-creator matching process. It serves as the first step in the UGC pipeline for product-related content.
"""

from __future__ import annotations
import sys
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any, Optional

# Load environment variables for API keys and other configurations.
load_dotenv()


def run_product_scrape_and_match():
    """
    Runs the complete product pipeline: scraping, filtering for new products,
    matching them with creator profiles, and saving the consolidated data.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from src.ugc.fastmoss.fastmoss_login import main as run_fastmoss_main

    print("--- Starting Product Scraping ---")
    all_products, new_products = run_fastmoss_main()

    if all_products is None:
        print("Product scraping failed or was aborted. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"--- Product Scraping Finished ---")
    print(f"Total products in database: {len(all_products)}")
    print(f"New products found: {len(new_products)}")

    # Identify all products that haven't been matched yet
    products_to_match = [p for p in all_products if 'matched_profile' not in p or p.get('matched_profile') is None]

    if not products_to_match:
        print("No new or unmatched products to process.")
    else:
        print(f"\n--- Starting Product-Creator Matching for {len(products_to_match)} products ---")
        matched_products = run_product_matching(script_dir, project_root, products_to_match)

        if matched_products:
            # Create a dictionary of matched products for efficient updating
            matched_map = {p['product_id']: p for p in matched_products}
            
            # Update the main product list with the new matched data
            for i, product in enumerate(all_products):
                if product['product_id'] in matched_map:
                    all_products[i] = matched_map[product['product_id']]
            
            print("--- Product-Creator Matching Finished ---")
        else:
            print("Product matching failed or returned no results.", file=sys.stderr)

    # --- Save Consolidated Data ---
    products_base_dir = os.path.join(project_root, 'outputs', 'products')
    final_output_path = os.path.join(products_base_dir, 'products_data.json')
    
    try:
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_products, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully saved {len(all_products)} products to '{final_output_path}'")
    except IOError as e:
        print(f"Error writing final data to file: {e}", file=sys.stderr)

    # --- Cleanup Old Matched File ---
    old_matched_file = os.path.join(products_base_dir, 'products_data_matched.json')
    if os.path.exists(old_matched_file):
        try:
            os.remove(old_matched_file)
            print(f"Successfully removed old matched file: '{old_matched_file}'")
        except OSError as e:
            print(f"Error removing old matched file: {e}", file=sys.stderr)


def load_json_file(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Loads and returns data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'", file=sys.stderr)
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing JSON file '{file_path}': {e}", file=sys.stderr)
        return None


def construct_matching_prompt(profiles: List[Dict[str, Any]], products: List[Dict[str, Any]]) -> str:
    """Constructs the prompt for the AI model to match products to profiles."""
    
    # Simplify profiles for the prompt to focus on relevant data.
    simplified_profiles = [
        {
            "name": p.get("name"),
            "tagline": p.get("tagline"),
            "niches": p.get("niches"),
        }
        for p in profiles
    ]

    # Simplify products to include only an index and description.
    simplified_products = [
        {
            "index": i,
            "description": p.get("Products", "No description available."),
        }
        for i, p in enumerate(products)
    ]

    return f"""
You are an expert talent manager. Your task is to match products to the most suitable UGC creator based on their profiles.

Analyze the creator profiles provided below, paying close attention to their niches, style, and audience.
Then, for each product in the product list, determine which creator is the best fit.

RULES:
1. You MUST evaluate every single product in the "PRODUCTS TO MATCH" list.
2. For each product, you must provide the name of the matched creator.
3. If and only if NO creator is a suitable match for a product, you must use the string "None".
4. For each match, you MUST provide a brief `explanation` (one sentence) for why the creator is a good fit for the product, based on their niches and style. For "None" matches, the explanation should be an empty string.
5. Your final output MUST be a single valid JSON object with one key, "matches", which contains a JSON array of objects. Each object in the array must contain 'product_index' (number), 'matched_creator_name' (string), and 'explanation' (string).
6. Do not include any other text, explanations, or formatting in your response. Ensure the entire response is a single, complete, and valid JSON object without any truncation.

### CREATOR PROFILES:
{json.dumps(simplified_profiles, indent=2)}

### PRODUCTS TO MATCH:
{json.dumps(simplified_products, indent=2)}

### EXAMPLE RESPONSE FORMAT:
{{
  "matches": [
    {{
      "product_index": 0,
      "matched_creator_name": "Emily Carter",
      "explanation": "This product fits perfectly into Emily's lifestyle and wellness niche."
    }},
    {{
      "product_index": 1,
      "matched_creator_name": "Emily Carter",
      "explanation": "As a trusted voice in beauty, Emily's recommendation for this product would be authentic."
    }},
    {{
      "product_index": 2,
      "matched_creator_name": "None",
      "explanation": ""
    }}
  ]
}}
"""


def get_matches_from_ai(prompt: str, max_retries: int = 3) -> Optional[List[Dict[str, Any]]]:
    """
    Sends the prompt to the AI model, gets the matching results, and retries on failure.
    This function now handles JSON parsing internally to validate the response before returning.
    """
    # API Configuration
    API_KEY = os.getenv("AIML_API_KEY")
    AIML_BASE_URL = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
    MODEL_ID = os.getenv("AIML_MODEL_ID", "gpt-5")

    if not API_KEY:
        print("Error: AIML_API_KEY is not set. Please set it in your .env file.", file=sys.stderr)
        return None

    # Initialize the OpenAI client to connect to the AIML API.
    aiml_client = OpenAI(api_key=API_KEY, base_url=AIML_BASE_URL)

    for attempt in range(max_retries):
        print(f"Sending request to AI for product matching (Attempt {attempt + 1}/{max_retries})...")
        try:
            response = aiml_client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = response.choices[0].message.content
            
            # Try to parse the JSON and validate its structure
            response_data = json.loads(content)
            if "matches" in response_data and isinstance(response_data["matches"], list):
                print("Successfully received and parsed valid response from AI.")
                return response_data["matches"]
            else:
                raise ValueError("AI response is missing the 'matches' key or it is not a list.")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Attempt {attempt + 1} failed. Could not parse AI response. Error: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print(f"Max retries reached. Failing.", file=sys.stderr)
                
        except Exception as e:
            print(f"An unexpected error occurred while calling the AI API: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print(f"Max retries reached. Failing.", file=sys.stderr)
                
    return None


def run_product_matching(script_dir: str, project_root: str, products_to_match: list[dict]) -> list[dict] | None:
    """
    Runs the product matching process for a given list of products.
    Returns the list of products with matching information added.
    """
    profiles_path = os.path.join(script_dir, 'profiles.json')
    profiles = load_json_file(profiles_path)
    if not profiles:
        return None

    # Process products in batches to avoid hitting API token limits
    batch_size = 5
    all_matches = []
    
    for i in range(0, len(products_to_match), batch_size):
        batch = products_to_match[i:i+batch_size]
        print(f"\n--- Processing Batch {i//batch_size + 1}/{(len(products_to_match) + batch_size - 1)//batch_size} (Products {i+1}-{i+len(batch)}) ---")

        prompt = construct_matching_prompt(profiles, batch)
        match_list = get_matches_from_ai(prompt)
        
        if not match_list:
            print(f"Warning: Failed to get matches for a batch after multiple retries. Skipping this batch.", file=sys.stderr)
            continue
        
        # Adjust product indices in the response to be absolute to the original `products_to_match` list
        for match in match_list:
            match['product_index'] += i
            
        all_matches.extend(match_list)
        
    if not all_matches:
        print("Aborting: No matches were successfully retrieved from any batch.", file=sys.stderr)
        return products_to_match # Return original list if matching fails
        
    # Create a mapping from product index to creator name and explanation
    match_map = {item['product_index']: {
        "name": item['matched_creator_name'],
        "explanation": item.get('explanation', '')
    } for item in all_matches}
    
    # Add the matched profile to each product in the input list
    for i, product in enumerate(products_to_match):
        match_info = match_map.get(i)
        if match_info:
            matched_name = match_info['name']
            product['matched_profile'] = None if matched_name == "None" else matched_name
            product['match_explanation'] = match_info['explanation']
        else:
            # Ensure keys exist even if matching failed for this item
            product['matched_profile'] = None
            product['match_explanation'] = ''
            
    return products_to_match


if __name__ == "__main__":
    run_product_scrape_and_match()
