import os
import time
import requests
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
aiml_api_key = os.getenv("AIML_API_KEY")
aiml_api_url = "https://api.aimlapi.com/v1/chat/completions"

# Read the style guide text from the file
try:
    with open("templates/fern_v1.txt", "r", encoding="utf-8") as f:
        style_guide = f.read()
except FileNotFoundError:
    style_guide = "Default style guide: investigative, data-driven, and tells a compelling story."

def generate_narration(idea, style_guide_text, api_type):
    """
    Generates a narration script using the specified API.
    """
    words_for_10_min = 10 * 150
    prompt = (
        f"Write a narration script for a 10-minute YouTube documentary about '{idea}'. "
        f"The script should be approximately {words_for_10_min} words long. "
        f"**CRITICAL: Output ONLY the words the narrator speaks - nothing else.** "
        f"Write ONLY what the narrator says aloud, as one continuous flowing narrative. "
        f"Study the example transcript carefully - notice how it's pure narration without any production elements. "
        f"Match that investigative, data-driven style that tells a compelling story with specific details and numbers. "
        f"Start with an immediate hook that grabs attention like the example does.\n\n"
        f"Example transcript style to follow:\n\n---\n{style_guide_text}\n---\n"
    )

    if api_type == "aiml":
        return call_aiml_api(prompt)
    elif api_type == "openai":
        return call_openai_api(prompt)

def generate_image_prompt(phrase, api_type):
    """
    Generates an image prompt using the specified API.
    """
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

    if api_type == "aiml":
        return call_aiml_api(prompt_for_flux)
    elif api_type == "openai":
        return call_openai_api(prompt_for_flux)

def call_aiml_api(prompt):
    if not aiml_api_key:
        return None, "AIML_API_KEY not found in .env file"
    headers = {"Authorization": f"Bearer {aiml_api_key}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(aiml_api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)

def call_openai_api(prompt):
    if not openai_api_key:
        return None, "OPENAI_API_KEY not found in .env file"
    client = openai.OpenAI(api_key=openai_api_key)
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)

if __name__ == "__main__":
    idea = "The Mysterious Disappearance of the Roanoke Colony"
    
    print("--- Testing aimlapi.com ---")
    start_time_aiml = time.time()
    narration_aiml, error_aiml_narration = generate_narration(idea, style_guide, "aiml")
    if error_aiml_narration:
        print(f"Error generating narration with aimlapi.com: {error_aiml_narration}")
    else:
        first_phrase_aiml = " ".join(narration_aiml.split()[:15])
        image_prompt_aiml, error_aiml_prompt = generate_image_prompt(first_phrase_aiml, "aiml")
        end_time_aiml = time.time()
        if error_aiml_prompt:
            print(f"Error generating image prompt with aimlapi.com: {error_aiml_prompt}")
        else:
            print(f"aimlapi.com took {end_time_aiml - start_time_aiml:.4f} seconds.")

    print("\n" + "-" * 20 + "\n")

    print("--- Testing OpenAI API ---")
    start_time_openai = time.time()
    narration_openai, error_openai_narration = generate_narration(idea, style_guide, "openai")
    if error_openai_narration:
        print(f"Error generating narration with OpenAI: {error_openai_narration}")
    else:
        first_phrase_openai = " ".join(narration_openai.split()[:15])
        image_prompt_openai, error_openai_prompt = generate_image_prompt(first_phrase_openai, "openai")
        end_time_openai = time.time()
        if error_openai_prompt:
            print(f"Error generating image prompt with OpenAI: {error_openai_prompt}")
        else:
            print(f"OpenAI API took {end_time_openai - start_time_openai:.4f} seconds.")
