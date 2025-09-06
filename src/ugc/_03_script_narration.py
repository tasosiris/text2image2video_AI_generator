"""
This script finds the latest UGC script JSON file, extracts the narration phrases,
and uses the text-to-speech tool to generate audio files for each phrase.
"""

import os
import json
import sys
import glob
import shutil
from typing import Optional

# Adjust the Python path to allow for relative imports from the parent 'src' directory.
# This makes the script more robust and independent of the execution environment.
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.tools.narration import narrate_phrases, NarrationSettings


def find_latest_script_json(base_dir: str) -> Optional[str]:
    """
    Finds the most recently modified .json file in a directory and its subdirectories.

    Args:
        base_dir: The directory to search within.

    Returns:
        The path to the latest JSON file, or None if no files are found.
    """
    # Create a search pattern to find all .json files recursively.
    search_pattern = os.path.join(base_dir, '**', '*.json')
    json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        return None
    
    # Return the file with the most recent modification time.
    latest_file = max(json_files, key=os.path.getmtime)
    return latest_file


def run_ugc_narration(script_path: str):
    """
    Runs the UGC narration process for a specific script file.
    
    Args:
        script_path: The full path to the UGC script JSON file to narrate.
    """
    # --- Configuration ---
    # Define the path to the voice sample you want to use for narration.
    VOICE_SAMPLE_PATH = os.path.join(project_root, 'templates', 'female_voice_new.mp3')

    print(f"Narrating script: {os.path.basename(script_path)}")

    # Check if the required voice sample file exists before proceeding.
    if not os.path.exists(VOICE_SAMPLE_PATH):
        print(f"Error: Voice sample not found at '{VOICE_SAMPLE_PATH}'.", file=sys.stderr)
        print("Please ensure the voice sample file exists to be used as a voice template.", file=sys.stderr)
        sys.exit(1)

    # Load the script data from the JSON file.
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            script_data = json.load(f)
        
        # Extract all 'narration' fields and add a trailing dot for a pause.
        raw_phrases = [scene.get('narration', '') for scene in script_data if scene.get('narration')]
        narration_phrases = []
        for phrase in raw_phrases:
            stripped_phrase = phrase.strip()
            # Add a dot only if it doesn't already end with sentence-ending punctuation.
            if not any(stripped_phrase.endswith(p) for p in ['.', '!', '?']):
                narration_phrases.append(stripped_phrase + '.')
            else:
                narration_phrases.append(stripped_phrase)
        
        if not narration_phrases:
            print("Error: The script contains no narration phrases to synthesize.", file=sys.stderr)
            sys.exit(1)

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading or parsing JSON file '{script_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Define the output directory for the generated narration MP3s.
    script_containing_folder = os.path.dirname(script_path)
    
    # The output folder for audio will be a subdirectory named 'audio'.
    output_dir_for_narration = os.path.join(script_containing_folder, 'audio')
    
    # Check if the output directory already exists. If so, delete it for a clean run.
    if os.path.exists(output_dir_for_narration):
        print(f"Output folder '{os.path.basename(output_dir_for_narration)}' already exists. Deleting and recreating it.")
        shutil.rmtree(output_dir_for_narration)

    # The narration tool will create the directory.
    print(f"Narration files will be saved in: {output_dir_for_narration}")

    # Set up the configuration for the text-to-speech model.
    narration_settings = NarrationSettings(
        device="auto",  # Automatically use CUDA if available, otherwise CPU.
        voice_sample_path=VOICE_SAMPLE_PATH,
        end_silence_ms=400,  # Add a small pause at the end of each clip.
        exaggeration=0.7,
        temperature=0.75,
        cfg_weight=0.3,
        repetition_penalty=1.5,
    )

    print(f"Starting narration for {len(narration_phrases)} phrases...")

    # Print each phrase that will be narrated for visibility.
    print("-" * 30)
    for i, phrase in enumerate(narration_phrases, 1):
        print(f"Narrating phrase {i}: {phrase}")
    print("-" * 30)

    # Call the narration tool to generate the audio files concurrently.
    narrate_phrases(
        phrases=narration_phrases,
        output_dir=output_dir_for_narration,
        settings=narration_settings,
        max_workers=3,  # Use up to 3 parallel threads for faster generation.
        filename_prefix="scene_",  # Name files like 'scene_001.mp3', 'scene_002.mp3'.
        start_index=1,
        create_narration_subdir=True, # Let the tool create the 'audio' subdir.
    )

    print("\nNarration generation complete.")

if __name__ == "__main__":
    # This block allows the script to be run standalone.
    # It finds the latest script and then calls the main narration function.
    UGC_SCRIPTS_DIR = os.path.join(project_root, 'outputs', 'ugc_scripts')

    print("Searching for the latest UGC script...")
    latest_script_path = find_latest_script_json(UGC_SCRIPTS_DIR)

    if latest_script_path:
        run_ugc_narration(latest_script_path)
    else:
        print("Error: No UGC script JSON files found in the outputs directory.", file=sys.stderr)
        sys.exit(1)
