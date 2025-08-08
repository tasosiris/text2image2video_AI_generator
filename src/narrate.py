import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, SpeakOptions
import re
import concurrent.futures

# Load environment variables from a .env file
load_dotenv()

# Get the Deepgram API key from environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Initialize the Deepgram client
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

def narrate_sentence(sentence, output_path):
    """
    Uses Deepgram to convert a single sentence to speech and save it as an MP3 file.
    """
    try:
        # Configure the request options
        options = SpeakOptions(
            model="aura-2-hyperion-en",
            encoding="mp3"
        )
        
        # Make the request to Deepgram's API
        response = deepgram.speak.rest.v("1").save(
            output_path,
            {"text": sentence},
            options
        )
        print(f"Successfully saved narration to '{output_path}'")
        return output_path
        
    except Exception as e:
        print(f"Error while processing sentence: {e}")
        return None

def narrate_phrases(phrases, output_dir):
    """
    Converts a list of phrases to speech and saves them as MP3 files.
    """
    # Create the narration subfolder
    narration_output_dir = os.path.join(output_dir, "narration")
    os.makedirs(narration_output_dir, exist_ok=True)
    
    if not phrases:
        print("No phrases provided to narrate.")
        return
        
    print(f"Narrating {len(phrases)} phrases.")

    # Create tasks for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, phrase in enumerate(phrases):
            # Format the filename with a leading zero for single-digit numbers
            filename = f"narration_{i+1:03d}.mp3"
            output_path = os.path.join(narration_output_dir, filename)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"  -> Narration for phrase {i+1} already exists. Skipping.")
                continue

            future = executor.submit(narrate_sentence, phrase, output_path)
            futures.append(future)
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")

    print("\nAll narration files have been generated.")
