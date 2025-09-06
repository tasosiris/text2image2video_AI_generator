import os
import json
import requests
import time
import sys
import subprocess

API_BASE = "http://127.0.0.1:8188"
DEFAULT_IMAGE = "product_1.jpeg"  # Default image name in ComfyUI
OUTPUT_NODE_ID = "79"  # SaveImage node that produces the final file
COMFYUI_START_COMMAND = r"C:\ComfyUISage\ComfyUI-Easy-Install\run_nvidia_gpu.bat"

def check_comfyui_connection(timeout: int = 10) -> bool:
    """Check if ComfyUI server is running and accessible."""
    try:
        response = requests.get(f"{API_BASE}/system_stats", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_comfyui_server():
    """Start ComfyUI server and wait for it to be ready."""
    print("ComfyUI server not running. Starting ComfyUI...")
    
    # Start the ComfyUI server in a new process
    try:
        # Start the batch file in a new command prompt window
        subprocess.Popen([COMFYUI_START_COMMAND], shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        print("ComfyUI server starting...")
        
        # Wait for the server to be ready
        max_wait_time = 120  # Maximum wait time in seconds
        wait_interval = 5    # Check every 5 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            print(f"Waiting for ComfyUI server to start... ({elapsed_time}s/{max_wait_time}s)")
            time.sleep(wait_interval)
            elapsed_time += wait_interval
            
            if check_comfyui_connection():
                print("✓ ComfyUI server is now running and ready!")
                return True
        
        print("✗ ComfyUI server failed to start within the timeout period.")
        return False
        
    except Exception as e:
        print(f"✗ Failed to start ComfyUI server: {e}")
        return False

def ensure_comfyui_running():
    """Ensure ComfyUI is running, start it if needed."""
    if check_comfyui_connection():
        print("✓ ComfyUI server is already running.")
        return True
    else:
        return start_comfyui_server()

def upload_image_to_comfy(filepath: str) -> str:
    """Uploads an image to the ComfyUI server's input directory."""
    with open(filepath, 'rb') as f:
        files = {'image': (os.path.basename(filepath), f, 'image/jpeg')}
        resp = requests.post(f"{API_BASE}/upload/image", files=files, data={'overwrite': 'true'})
        resp.raise_for_status()
        return resp.json()['name']

def get_image_data(prompt_id: str, timeout: int = 600) -> bytes | None:
    """Polls for and retrieves the final image from a workflow."""
    start_time = time.time()
    print("Processing image...")
    while time.time() - start_time < timeout:
        try:
            res = requests.get(f"{API_BASE}/history/{prompt_id}")
            res.raise_for_status()
            history = res.json()
            if prompt_id in history and 'outputs' in history[prompt_id]:
                outputs = history[prompt_id]['outputs']
                if OUTPUT_NODE_ID in outputs and 'images' in outputs[OUTPUT_NODE_ID]:
                    image_info = outputs[OUTPUT_NODE_ID]['images'][0]
                    img_resp = requests.get(f"{API_BASE}/view?filename={image_info['filename']}&subfolder={image_info.get('subfolder', '')}&type={image_info.get('type', 'output')}")
                    img_resp.raise_for_status()
                    return img_resp.content
        except requests.exceptions.RequestException as e:
            print(f"Polling failed for prompt {prompt_id}: {e}", file=sys.stderr)
        
        # Progress indicator with longer wait time
        elapsed = int(time.time() - start_time)
        sys.stdout.write(f"\rWaiting for result... {elapsed}s")
        sys.stdout.flush()
        time.sleep(15)  # Wait 15 seconds between checks instead of 2
        
    raise TimeoutError(f"\nTimed out waiting for image from prompt {prompt_id}")

def _send_and_wait_for_image(payload, output_dir, output_filename):
    """Sends a payload to ComfyUI and waits for the image result."""
    # Send the API request
    response = requests.post(f"{API_BASE}/prompt", json=payload)
    response.raise_for_status()
    
    result = response.json()
    prompt_id = result.get("prompt_id")
    print(f"Request sent successfully. Prompt ID: {prompt_id}")
    
    # Always wait for the image to be generated
    if prompt_id:
        try:
            image_bytes = get_image_data(prompt_id)
            if image_bytes:
                # If we have an output directory, save the image
                if output_dir:
                    # Create media directory
                    media_dir = os.path.join(output_dir, "media")
                    os.makedirs(media_dir, exist_ok=True)
                    
                    # Save the image with the specified filename
                    output_path = os.path.join(media_dir, output_filename)
                    
                    with open(output_path, "wb") as f:
                        f.write(image_bytes)
                    print(f"\n✓ Successfully saved edited image to: {output_path}")
                    return output_path
                else:
                    print(f"\n✓ Image generated successfully (not saved - no output directory provided)")
                    return image_bytes
            else:
                print("\nFailed to retrieve image data from ComfyUI.", file=sys.stderr)
        except TimeoutError as e:
            print(e, file=sys.stderr)
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
    
    return result

def send_comfy_request(workflow_file, image_path=None, output_dir=None, custom_prompt=None, output_filename="edited_product.png", image_node_id="109"):
    """Load a workflow JSON file and send it to ComfyUI."""
    # Load the JSON workflow
    script_dir = os.path.dirname(os.path.realpath(__file__))
    workflow_path = os.path.join(script_dir, workflow_file)
    
    with open(workflow_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # If a custom prompt is provided, update the workflow
    if custom_prompt:
        workflow["97"]["inputs"]["prompt"] = custom_prompt
        print(f"Using custom prompt: {custom_prompt[:100]}...")
    
    # If an image path is provided, upload it and update the workflow
    if image_path and os.path.exists(image_path):
        print(f"Uploading image: {image_path}")
        uploaded_filename = upload_image_to_comfy(image_path)
        print(f"Image uploaded as: {uploaded_filename}")
        
        # Update the workflow with the uploaded image
        workflow[image_node_id]["inputs"]["image"] = uploaded_filename
    else:
        # Use default image name
        print(f"Using default image: {DEFAULT_IMAGE}")
        workflow[image_node_id]["inputs"]["image"] = DEFAULT_IMAGE
    
    payload = {"prompt": workflow}
    return _send_and_wait_for_image(payload, output_dir, output_filename)

def send_hand_merge_request(workflow_file, hand_image_path, product_image_path, output_dir, output_filename):
    """Loads a workflow to merge a product into a hand image and sends it to ComfyUI."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    workflow_path = os.path.join(script_dir, workflow_file)
    
    with open(workflow_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    # Upload hand image and update workflow node 110
    if hand_image_path and os.path.exists(hand_image_path):
        print(f"Uploading hand image: {os.path.basename(hand_image_path)}")
        uploaded_hand_filename = upload_image_to_comfy(hand_image_path)
        print(f"Hand image uploaded as: {uploaded_hand_filename}")
        workflow["110"]["inputs"]["image"] = uploaded_hand_filename
    else:
        print(f"Error: Hand image not found at {hand_image_path}", file=sys.stderr)
        return None

    # Upload product image and update workflow node 113
    if product_image_path and os.path.exists(product_image_path):
        print(f"Uploading product image: {os.path.basename(product_image_path)}")
        uploaded_product_filename = upload_image_to_comfy(product_image_path)
        print(f"Product image uploaded as: {uploaded_product_filename}")
        workflow["113"]["inputs"]["image"] = uploaded_product_filename
    else:
        print(f"Error: Product image not found at {product_image_path}", file=sys.stderr)
        return None
        
    payload = {"prompt": workflow}
    return _send_and_wait_for_image(payload, output_dir, output_filename)

def run_image_editing_for_pipeline(product_image_filename: str, script_dir: str):
    """Function called by ugc_pipeline.py to edit a specific product image using the new 4-step workflow."""
    import os
    
    # Ensure ComfyUI is running before proceeding
    if not ensure_comfyui_running():
        print("Error: ComfyUI server could not be started. Image editing cannot proceed.")
        return False
    
    # Get the project root (assuming this script is in src/ugc/)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Construct the full path to the product image
    # product_image_filename contains path like "Beauty/photos/beauty_123.jpeg"
    product_image_path = os.path.normpath(os.path.join(project_root, "outputs", "products", product_image_filename))
    
    # Create the media folder inside the script directory
    media_dir = os.path.join(script_dir, "media")
    
    print(f"Running image editing for: {os.path.basename(product_image_filename)}")
    print(f"Product image path: {product_image_path}")
    print(f"Output will be saved to: {media_dir}")
    
    if not os.path.exists(product_image_path):
        print(f"Error: Product image not found at {product_image_path}")
        return False
    
    success_count = 0
    
    # Step 1: First extraction pass using qwen_product_extractor
    print("\n--- Step 1: First extraction pass ---")
    result_step1 = send_comfy_request(
        "qwen_product_extractor.json", 
        product_image_path, 
        script_dir, 
        custom_prompt=None,  # Don't change the prompt in extractor
        output_filename="step_1.png",
        image_node_id="96"  # LoadImage node in extractor workflow
    )
    
    if isinstance(result_step1, str):  # If it returns a file path, it was successful
        success_count += 1
        print("✓ Step 1 extraction completed successfully")
        step1_path = result_step1
    else:
        print("✗ Step 1 extraction failed")
        return False
    
    # Step 2: Second extraction pass using qwen_product_extractor on step 1 result
    print("\n--- Step 2: Second extraction pass ---")
    result_step2 = send_comfy_request(
        "qwen_product_extractor.json", 
        step1_path, 
        script_dir, 
        custom_prompt=None,  # Don't change the prompt in extractor
        output_filename="step_2.png",
        image_node_id="96"  # LoadImage node in extractor workflow
    )
    
    if isinstance(result_step2, str):  # If it returns a file path, it was successful
        success_count += 1
        print("✓ Step 2 extraction completed successfully")
        step2_path = result_step2
    else:
        print("✗ Step 2 extraction failed")
        return False
    
    # Step 3: Run final extracted product through qwen_product_tiktok (background version)
    print("\n--- Step 3: Processing final product with background ---")
    prompt1 = "The product is the central focus, resting on a clean wooden kitchen surface. The background is a warm, slightly blurred kitchen setting. Shot in a natural phone photo style, with cinematic quality and soft, appealing lighting."
    
    result1 = send_comfy_request(
        "qwen_product_tiktok.json",
        step2_path,
        script_dir,
        custom_prompt=prompt1,
        output_filename="edited_product_1.png",
        image_node_id="109"  # LoadImage node in tiktok workflow
    )
    
    if isinstance(result1, str):  # If it returns a file path, it was successful
        success_count += 1
        print("✓ Background version completed successfully")
    else:
        print("✗ Background version failed")
    
    # Step 4: Run final extracted product through qwen_product_tiktok (hand version)
    print("\n--- Step 4: Processing final product with hand ---")
    
    hand_image_for_merge = os.path.join(project_root, "src", "ugc", "fluxkrea_00002_.png")
    
    result2 = send_hand_merge_request(
        "qwen_hand_product_merge.json",
        hand_image_for_merge,
        step2_path,
        script_dir,
        "edited_product_2.png"
    )
    
    if isinstance(result2, str):  # If it returns a file path, it was successful
        success_count += 1
        print("✓ Hand version completed successfully")
    else:
        print("✗ Hand version failed")
    
    print(f"\n--- Image editing completed: {success_count}/4 steps successful ---")
    return success_count >= 3  # Return True if at least the two extractions and one final image were successful

def run_image_editing(input_image_path: str, output_dir: str):
    """Function for direct usage with specific image path and output directory."""
    print(f"Running image editing for: {input_image_path}")
    print(f"Output will be saved to: {output_dir}")
    return send_comfy_request("qwen_product_tiktok.json", input_image_path, output_dir)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments - use default image, no saving
        send_comfy_request("qwen_product_tiktok.json")
    elif len(sys.argv) == 2:
        # One argument - output directory only (use default image)
        output_dir = sys.argv[1]
        send_comfy_request("qwen_product_tiktok.json", output_dir=output_dir)
    elif len(sys.argv) == 3:
        # Two arguments - image path and output directory
        image_path = sys.argv[1]
        output_dir = sys.argv[2]
        send_comfy_request("qwen_product_tiktok.json", image_path, output_dir)
    else:
        print("Usage:")
        print("  python _02_edit_image.py                           # Use default image, no saving")
        print("  python _02_edit_image.py <output_dir>              # Use default image, save to output_dir")
        print("  python _02_edit_image.py <image_path> <output_dir> # Use specific image, save to output_dir")
        sys.exit(1)
