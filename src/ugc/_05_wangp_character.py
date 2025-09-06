import time
import os
import requests
import subprocess
from playwright.sync_api import sync_playwright
import json
from mutagen.mp3 import MP3
from typing import Optional
import sys

# --- Configuration ---
URL = "http://localhost:7860"
JSON_FILE_TO_UGC = r"C:\Users\tasos\Downloads\UGC_defaults.json"
# The character photo is now a constant
CHARACTER_IMAGE_PATH = r"C:\Users\tasos\Code\Flux_Dev\outputs\ugc_scripts\Emily_Carter\photo_1.png"
FRAME_RATE = 25
WAN2GP_DIR = r"C:\Users\tasos\Code\HunyuanVideo\Wan2GP"


def check_localhost_ready(url, timeout=5):
    """Check if the localhost service is ready."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
        return False

def wait_for_video_completion(output_folder=r"C:\Users\tasos\Code\HunyuanVideo\Wan2GP\outputs", timeout_minutes=40, check_interval=20):
    """
    Wait for video generation to complete by monitoring the output folder.
    
    Args:
        output_folder: Path to the WanGP outputs folder
        timeout_minutes: Maximum time to wait in minutes (default: 20)
        check_interval: Time between checks in seconds (default: 20)
    
    Returns:
        bool: True if new files are detected, False if timeout
    """
    print(f"Monitoring output folder: {output_folder}")
    print(f"Timeout: {timeout_minutes} minutes, Check interval: {check_interval} seconds")
    
    # Get initial file count
    try:
        initial_files = set()
        if os.path.exists(output_folder):
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    initial_files.add(os.path.join(root, file))
        initial_count = len(initial_files)
        print(f"Initial file count: {initial_count}")
    except Exception as e:
        print(f"Error reading initial folder state: {e}")
        initial_files = set()
        initial_count = 0
    
    # Wait and check for new files
    timeout_seconds = timeout_minutes * 60
    elapsed_time = 0
    
    while elapsed_time < timeout_seconds:
        time.sleep(check_interval)
        elapsed_time += check_interval
        
        try:
            current_files = set()
            if os.path.exists(output_folder):
                for root, dirs, files in os.walk(output_folder):
                    for file in files:
                        current_files.add(os.path.join(root, file))
            
            current_count = len(current_files)
            new_files = current_files - initial_files
            
            # If we have new files, consider it complete
            if new_files:
                print(f"\nNew files detected:")
                for new_file in sorted(new_files):
                    print(f"  - {os.path.basename(new_file)}")
                return True
                
        except Exception as e:
            print(f"\nError checking folder: {e}")
        
        # Show progress with carriage return to update in place
        remaining_minutes = (timeout_seconds - elapsed_time) // 60
        remaining_seconds = (timeout_seconds - elapsed_time) % 60
        import sys
        sys.stdout.write(f"\rCheck {elapsed_time//check_interval}: {current_count} files ({len(new_files)} new) | Remaining: {remaining_minutes}m {remaining_seconds}s")
        sys.stdout.flush()
    
    print("Timeout reached - no new files detected")
    return False

def start_wan2gp_service():
    """Start the Wan2GP service by opening a terminal and running the commands."""
    print("\n[Service Startup: Starting Wan2GP Service]")
    print(f"Action: Service not detected at {URL}")
    print(f"Action: Opening terminal in '{WAN2GP_DIR}'...")
    
    # Create the command to run in a new terminal
    # Using PowerShell to open a new window, activate conda environment, and run the service
    escaped_dir = WAN2GP_DIR.replace("\\", "\\\\")
    command = f"Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd \"{escaped_dir}\"; conda activate wan2gp; python wgp.py --i2v'"
    
    try:
        # Execute the command using PowerShell
        subprocess.run(["powershell", "-Command", command], check=True)
        print("SUCCESS: Terminal opened with Wan2GP service starting...")
        
        # Wait for the service to start up
        print("Action: Waiting for service to become ready...")
        max_wait_time = 120  # 2 minutes timeout
        wait_interval = 5
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            if check_localhost_ready(URL):
                print(f"SUCCESS: Service is ready at {URL}")
                return True
            
            print(f"Info: Service not ready yet, waiting... ({elapsed_time}/{max_wait_time}s)")
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        print(f"WARNING: Service did not become ready within {max_wait_time} seconds")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to start service: {e}")
        return False

def run_wangp_character_generation(script_dir: str):
    """
    Main function to generate character videos based on the UGC script.
    It reads the script, filters for UGC talking scenes, and generates a video for each.
    """
    # --- Step 1: Load the script JSON ---
    json_path = ""
    for file in os.listdir(script_dir):
        if file.endswith("_script.json"):
            json_path = os.path.join(script_dir, file)
            break
    
    if not json_path:
        print(f"Error: Could not find a '_script.json' file in {script_dir}")
        return False
        
    with open(json_path, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    # --- Step 2: Filter for 'UGC talking' scenes ---
    ugc_scenes = [
        scene for scene in script_data
        if scene.get("scene_type", "").lower() == "ugc talking"
    ]
    
    if not ugc_scenes:
        print("No 'UGC talking' scenes found in the script. Nothing to generate.")
        return True # Not an error, just no work to do.

    print(f"Found {len(ugc_scenes)} 'UGC talking' scenes to process.")
    
    success_count = 0
    
    # --- Step 3: Process each UGC scene ---
    for i, scene in enumerate(ugc_scenes, 1):
        scene_number = scene.get("scene_number", i)
        print(f"\n--- Processing Scene {scene_number}/{len(ugc_scenes)} ---")

        # Construct the path to the corresponding audio file
        audio_filename = f"scene_{scene_number:03d}.mp3"
        audio_file_path = os.path.join(script_dir, "audio", "narration", audio_filename)
        
        if not os.path.exists(audio_file_path):
            print(f"✗ ERROR: Audio file not found for scene {scene_number}: {audio_file_path}")
            continue

        if process_single_scene(audio_file_path, scene.get("AI Video Prompt", "")):
            success_count += 1
            print(f"✓ Scene {scene_number} processed successfully")
        else:
            print(f"✗ Scene {scene_number} failed to process")

    print(f"\n--- WanGP Character processing completed: {success_count}/{len(ugc_scenes)} scenes successful ---")
    return success_count > 0

def process_single_scene(audio_file_path: str, prompt_text: str):
    """Process a single scene by generating a video with the character and narration."""
    
    print("--- Automated UGC Character Generation ---")
    print(f"Character Image: {os.path.basename(CHARACTER_IMAGE_PATH)}")
    print(f"Audio File: {os.path.basename(audio_file_path)}")
    print(f"Prompt: {prompt_text}")

    with sync_playwright() as p:
        try:
            # --- Step 1: Calculate Required Frames ---
            print("\n[Step 1: Calculate Required Frames]")
            audio = MP3(audio_file_path)
            duration_seconds = audio.info.length
            exact_frames = duration_seconds * FRAME_RATE
            target_dim = exact_frames - 1
            adjusted_dim = round(target_dim / 4) * 4
            num_frames = adjusted_dim + 1
            print(f"SUCCESS: Audio duration {duration_seconds:.2f}s -> {num_frames} frames.")

            # --- Step 2: Check if Service is Ready ---
            print("\n[Step 2: Check Service Status]")
            if not check_localhost_ready(URL):
                print("Info: Service not ready, starting Wan2GP service...")
                if not start_wan2gp_service():
                    raise Exception("Failed to start Wan2GP service")
            else:
                print("SUCCESS: Service is already running.")

            # --- Step 3: Browser Automation ---
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            
            print("\n[Step 3: Navigate to Page]")
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            print("SUCCESS: Page loaded.")

            print("\n[Step 4: Upload JSON File]")
            json_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[20]/div[1]/button/input")
            json_input.set_input_files(JSON_FILE_TO_UGC)
            print("SUCCESS: JSON file selected.")
            time.sleep(3)

            print("\n[Step 5: Upload Image File]")
            image_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[6]/div[10]/button/input")
            image_input.set_input_files(CHARACTER_IMAGE_PATH)
            print("SUCCESS: Character image selected.")

            print("\n[Step 6: Upload Audio File]")
            audio_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[8]/div[1]/div[2]/button/input")
            audio_input.set_input_files(audio_file_path)
            print("SUCCESS: Audio file selected.")
            time.sleep(3)

            print("\n[Step 7: Input Frame Count]")
            frames_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[15]/div/div[2]/div[2]/div[1]/div/input")
            frames_input.fill(str(num_frames))
            print(f"SUCCESS: Frame count '{num_frames}' entered.")

            # --- Prompting is temporarily disabled for testing ---
            # print("\n[Step 8: Set AI Video Prompt]")
            # prompt_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[10]/div/div/label/div/textarea")
            
            # # Use a more robust, human-like method for inputting text
            # print("Action: Clicking the text area to focus...")
            # prompt_input.click()
            # print("Action: Clearing any existing text...")
            # prompt_input.clear()
            # print("Action: Typing the prompt text sequentially...")
            # prompt_input.press_sequentially(prompt_text, delay=10) # Small delay to mimic human typing

            # print("SUCCESS: Prompt set.")

            # # Verify that the prompt was filled correctly
            # filled_text = prompt_input.input_value()
            # if filled_text == prompt_text:
            #     print("SUCCESS: Prompt content verified.")
            # else:
            #     print("WARNING: Prompt content mismatch after filling.")
            #     print(f"  - Expected: {prompt_text}")
            #     print(f"  - Found: {filled_text}")

            # print("Action: Waiting 1 second before clicking generate...")
            # time.sleep(1)

            print("\n[Step 8: Click Generate Button]")
            generate_button = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[2]/button[1]")
            generate_button.click()
            print("SUCCESS: Generate button clicked.")

            print("\n[Step 9: Wait for Generation to Complete]")
            if wait_for_video_completion():
                print("SUCCESS: Video generation completed!")
                return True
            else:
                print("WARNING: Video generation timed out or failed.")
                return False

        except Exception as e:
            print(f"\n--- AN ERROR OCCURRED ---")
            print(e)
            print("\nKeeping browser open for 30 seconds for inspection...")
            time.sleep(30)
            return False
            
        finally:
            if 'browser' in locals() and browser.is_connected():
                print("\n--- Script Finished ---")
                print("Closing browser.")
                browser.close()

def find_latest_run_directory() -> Optional[str]:
    """Finds the most recently modified subdirectory in the UGC scripts output folder."""
    try:
        # Determine the project's root directory to correctly place the 'outputs' folder.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        base_dir = os.path.join(project_root, 'outputs', 'ugc_scripts', 'Emily_Carter')

        if not os.path.isdir(base_dir):
            print(f"Error: Base directory not found at '{base_dir}'", file=sys.stderr)
            return None

        sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        if not sub_dirs:
            print(f"No subdirectories found in '{base_dir}'", file=sys.stderr)
            return None
            
        latest_dir = max(sub_dirs, key=os.path.getmtime)
        print(f"Found latest run directory: {os.path.basename(latest_dir)}")
        return latest_dir

    except Exception as e:
        print(f"Error finding latest run directory: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    # For standalone testing - find the latest run directory automatically
    latest_run_dir = find_latest_run_directory()
    if latest_run_dir:
        run_wangp_character_generation(latest_run_dir)
    else:
        print("Could not determine the latest run directory. Aborting.", file=sys.stderr)
        sys.exit(1)
