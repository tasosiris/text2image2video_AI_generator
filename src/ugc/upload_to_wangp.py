import time
import os
from playwright.sync_api import sync_playwright
from mutagen.mp3 import MP3

# --- Configuration ---
URL = "http://localhost:7860"
JSON_FILE_TO_UGC = r"C:\Users\tasos\Downloads\UGC_defaults.json"
IMAGE_FILE_TO_UPLOAD = r"C:\Users\tasos\Code\Flux_Dev\outputs\ugc_scripts\Emily_Carter\photo_1.png"
FRAME_RATE = 25

print("--- Automated Generation Script ---")
print(f"Target URL: {URL}")
print(f"JSON File: {JSON_FILE_TO_UGC}")
print(f"Image File: {IMAGE_FILE_TO_UPLOAD}")

with sync_playwright() as p:
    try:
        # --- Step 1: Find and Analyze Audio File ---
        print("\n[Step 1: Find Latest Audio File]")
        base_dir = r"outputs\ugc_scripts\Emily_Carter"
        print(f"Action: Searching for the latest directory in '{base_dir}'...")
        
        sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not sub_dirs:
            raise FileNotFoundError(f"No subdirectories found in '{base_dir}'")
            
        latest_dir = max(sub_dirs, key=os.path.getmtime)
        audio_file_path = os.path.join(latest_dir, "audio", "narration", "scene_001.mp3")
        
        print(f"SUCCESS: Found latest directory: {os.path.basename(latest_dir)}")
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file 'scene_001.mp3' not found in '{os.path.join(latest_dir, 'audio', 'narration')}'")
        print(f"SUCCESS: Found audio file to upload: {audio_file_path}")

        print("\n[Step 1a: Calculate Required Frames]")
        print(f"Action: Reading duration of '{os.path.basename(audio_file_path)}'...")
        audio = MP3(audio_file_path)
        duration_seconds = audio.info.length
        print(f"SUCCESS: Audio duration is {duration_seconds:.4f} seconds.")
        
        print(f"Action: Calculating frames at {FRAME_RATE} FPS...")
        exact_frames = duration_seconds * FRAME_RATE
        
        print(f"Info: Exact frames needed: {duration_seconds:.4f}s * {FRAME_RATE} fps = {exact_frames:.4f} frames.")
        
        print("Action: Adjusting frames for model compatibility (must be multiple of 4 after subtracting 1)...")
        # The model requires (num_frames - 1) to be divisible by 4.
        # So, we find the nearest multiple of 4 to (exact_frames - 1), then add 1 back.
        target_dim = exact_frames - 1
        adjusted_dim = round(target_dim / 4) * 4
        num_frames = adjusted_dim + 1

        print(f"Calculation: round(({exact_frames:.4f} - 1) / 4) * 4 + 1 = {num_frames}")
        print(f"SUCCESS: Adjusted to {num_frames} total frames.")

        # --- Step 2: Browser Automation ---
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        print("\n[Step 2: Navigate to Page]")
        print(f"Action: Navigating to {URL}...")
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        print("SUCCESS: Page loaded.")

        print("\n[Step 3: Upload JSON File]")
        json_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[20]/div[1]/button/input")
        print(f"Action: Uploading '{JSON_FILE_TO_UGC}'...")
        json_input.set_input_files(JSON_FILE_TO_UGC)
        print("SUCCESS: JSON file selected.")

        print("\n[Step 3a: Wait]")
        print("Action: Waiting 3 seconds...")
        time.sleep(3)

        print("\n[Step 4: Upload Image File]")
        image_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[6]/div[10]/button/input")
        print(f"Action: Uploading '{IMAGE_FILE_TO_UPLOAD}'...")
        image_input.set_input_files(IMAGE_FILE_TO_UPLOAD)
        print("SUCCESS: Image file selected.")

        print("\n[Step 5: Upload Audio File]")
        audio_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[8]/div[1]/div[2]/button/input")
        print(f"Action: Uploading '{audio_file_path}'...")
        audio_input.set_input_files(audio_file_path)
        print("SUCCESS: Audio file selected.")

        print("\n[Step 5a: Wait]")
        print("Action: Waiting 3 seconds...")
        time.sleep(3)

        print("\n[Step 6: Input Frame Count]")
        frames_input_selector = "xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[15]/div/div[2]/div[2]/div[1]/div/input"
        print(f"Action: Locating the frame count input with selector:\n'{frames_input_selector}'")
        frames_input = page.locator(frames_input_selector)
        print(f"Action: Filling the input with the calculated value: {num_frames}")
        frames_input.fill(str(num_frames))
        print("SUCCESS: Frame count has been entered.")

        print("\n[Step 6a: Wait Before Generate]")
        print("Action: Waiting 1 second...")
        time.sleep(1)

        print("\n[Step 7: Click Generate Button]")
        generate_button = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[2]/button[1]")
        print("Action: Clicking the generate button...")
        generate_button.click()
        print("SUCCESS: Generate button clicked.")

        print("\n[Step 8: Wait for Generation to Complete]")
        output_video_dir = r"C:\Users\tasos\Code\HunyuanVideo\Wan2GP\outputs"
        print(f"Action: Monitoring output directory for a new .mp4 file:\n'{output_video_dir}'")
        print("Info: This will wait indefinitely, checking every 5 seconds.")
        
        generation_start_time = time.time()
        
        while True:
            found_new_video = False
            for filename in os.listdir(output_video_dir):
                if filename.endswith('.mp4'):
                    filepath = os.path.join(output_video_dir, filename)
                    try:
                        # Check if file was created after we clicked 'Generate'
                        if os.path.getctime(filepath) > generation_start_time:
                            print(f"\nSUCCESS: New video generated: {filename}")
                            found_new_video = True
                            break
                    except FileNotFoundError:
                        # This can happen in a race condition if file is deleted while checking
                        continue
            
            if found_new_video:
                break
            
            time.sleep(5)

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(e)
        print("\nKeeping browser open for 30 seconds for inspection...")
        time.sleep(30)
        
    finally:
        if 'browser' in locals() and browser.is_connected():
            print("\n--- Script Finished ---")
            print("Closing browser.")
            browser.close()
