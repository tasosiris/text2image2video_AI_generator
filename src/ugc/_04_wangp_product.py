import time
import os
import requests
import subprocess
from playwright.sync_api import sync_playwright

# --- Configuration ---
URL = "http://localhost:7860"
JSON_FILE_TO_UGC = r"C:\Users\tasos\Downloads\UGC_Product_defaults.json"
WAN2GP_DIR = r"C:\Users\tasos\Code\HunyuanVideo\Wan2GP"

def check_localhost_ready(url, timeout=5):
    """Check if the localhost service is ready."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
        return False

def wait_for_video_completion(output_folder=r"C:\Users\tasos\Code\HunyuanVideo\Wan2GP\outputs", timeout_minutes=20, check_interval=20):
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

def run_wangp_product_generation(script_dir: str):
    """Function called by ugc_pipeline.py to generate videos from edited product images."""
    
    # Find the edited product images in the media folder
    media_dir = os.path.join(script_dir, "media")
    
    if not os.path.exists(media_dir):
        print(f"Error: Media directory not found at {media_dir}")
        return False
    
    # Look for the edited product images
    image_files = []
    for filename in ["edited_product_1.png", "edited_product_2.png"]:
        image_path = os.path.join(media_dir, filename)
        if os.path.exists(image_path):
            image_files.append(image_path)
    
    if not image_files:
        print(f"Error: No edited product images found in {media_dir}")
        return False
    
    print(f"Found {len(image_files)} edited product images to process")
    
    success_count = 0
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"\n--- Processing Image {i}/{len(image_files)}: {os.path.basename(image_file)} ---")
        
        if process_single_image(image_file):
            success_count += 1
            print(f"✓ Image {i} processed successfully")
        else:
            print(f"✗ Image {i} failed to process")
    
    print(f"\n--- WanGP processing completed: {success_count}/{len(image_files)} images successful ---")
    return success_count > 0  # Return True if at least one image was successful

def process_single_image(image_file_path: str):
    """Process a single image through the WanGP service."""
    print("--- Automated Product Image Generation Script ---")
    print(f"Target URL: {URL}")
    print(f"JSON File: {JSON_FILE_TO_UGC}")
    print(f"Image File: {image_file_path}")

    with sync_playwright() as p:
        try:
            # --- Step 1: Check if Service is Ready ---
            print("\n[Step 1: Check Service Status]")
            print(f"Action: Checking if service is ready at {URL}...")
            
            if not check_localhost_ready(URL):
                print("Info: Service not ready, starting Wan2GP service...")
                if not start_wan2gp_service():
                    raise Exception("Failed to start Wan2GP service")
            else:
                print("SUCCESS: Service is already running and ready.")

            # --- Step 2: Browser Automation ---
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            
            print("\n[Step 2: Navigate to Page]")
            print(f"Action: Navigating to {URL}...")
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            print("SUCCESS: Page loaded.")

            print("\n[Step 3: Upload JSON File]")
            json_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[20]/div[1]/button/input")
            print(f"Action: Uploading '{os.path.basename(JSON_FILE_TO_UGC)}'...")
            json_input.set_input_files(JSON_FILE_TO_UGC)
            print("SUCCESS: JSON file selected.")

            print("\n[Step 3a: Wait]")
            print("Action: Waiting 3 seconds...")
            time.sleep(3)

            print("\n[Step 4: Upload Image File]")
            image_input = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[5]/div[2]/button/input")
            print(f"Action: Uploading '{os.path.basename(image_file_path)}'...")
            image_input.set_input_files(image_file_path)
            print("SUCCESS: Image file selected.")

            print("\n[Step 4b: Set Universal Prompt]")
            prompt_text_area = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[1]/div[10]/div/div/label/div/textarea")
            universal_prompt = "The product is the main subject, perfectly still and in sharp focus. The camera has a gentle, natural motion, like a slow pan or orbit. The background is slightly blurred and out of focus."
            print(f"Action: Setting prompt to: '{universal_prompt}'")
            prompt_text_area.fill(universal_prompt)
            print("SUCCESS: Prompt set.")

            print("\n[Step 4a: Wait Before Generate]")
            print("Action: Waiting 1 second...")
            time.sleep(1)

            print("\n[Step 5: Click Generate Button]")
            generate_button = page.locator("xpath=/html/body/gradio-app/div/main/div[1]/div/div/div[2]/div[2]/div/div[3]/div/div[2]/button[1]")
            print("Action: Clicking the generate button...")
            generate_button.click()
            print("SUCCESS: Generate button clicked.")

            print("\n[Step 6: Generation in progress]")
            print("Info: Waiting for video generation to complete...")
            
            # Monitor the output folder for completion
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

if __name__ == "__main__":
    # For standalone testing - use a hardcoded path
    test_script_dir = r"C:\Users\tasos\Code\Flux_Dev\outputs\ugc_scripts\Emily_Carter\2025-09-01_health_ec7ddbbdc3ff"
    run_wangp_product_generation(test_script_dir)
