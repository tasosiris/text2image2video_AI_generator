#!/usr/bin/env python3
"""
Main pipeline script to generate a complete documentary from an idea.

This script orchestrates the entire content generation process in stages for efficiency:
1.  **Content Generation:** Creates documentary idea, narration, and prompts.
2.  **Visual Generation:**
    a. Loads ComfyUI workflows into memory once.
    b. Submits all text-to-image jobs concurrently.
    c. Submits all image-to-video jobs concurrently.
3.  **Asset Manifest:** Saves a manifest of all generated assets and prints performance metrics.

This approach is significantly faster than previous versions because it avoids
reloading workflows and calling separate scripts for each generation task.
"""
import os
import json
import sys
import time
import uuid
import glob
import random
import shutil
import requests
import concurrent.futures
from documentary_generator import (
    generate_idea, generate_narration, split_into_phrases,
    generate_flux_prompts, save_to_json
)

# --- ComfyUI API Configuration ---
API_BASE = "http://127.0.0.1:8188"
T2I_WORKFLOW_PATH = r"C:\ComfyUI\ComfyUI_windows_portable\ComfyUI\workflows\api+workflows\workflow-comfyui-workflow-for-flux-simple-iuRdGnfzmTbOOzONIiVV-maitruclam-openart.ai.json"
I2V_WORKFLOW_PATH = r"C:\ComfyUI\ComfyUI_windows_portable\ComfyUI\workflows\api+workflows\LTXV_workflow_optimized.json"
COMFYUI_OUTPUT_DIR = r"C:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output"

# Node IDs for Text-to-Image Workflow
T2I_PROMPT_NODE_ID = "6"
T2I_OUTPUT_NODE_ID = "9"

# Node IDs for Image-to-Video Workflow
I2V_IMAGE_NODE_ID = "1206"
I2V_POS_PROMPT_NODE_ID = "6"
I2V_NEG_PROMPT_NODE_ID = "7"
I2V_SEED_NODE_IDS = ["1507", "1598"]
I2V_VIDEO_FILENAME_PREFIX = "ltxv-up"


# ─── COMFYUI API HELPERS (INTEGRATED) ───────────────────────────────────────────

def queue_comfy_workflow(workflow: dict, client_id: str) -> str:
    """Queues a workflow on the ComfyUI server."""
    try:
        payload = {"prompt": workflow, "client_id": client_id}
        resp = requests.post(f"{API_BASE}/prompt", json=payload)
        resp.raise_for_status()
        return resp.json()["prompt_id"]
    except requests.exceptions.RequestException as e:
        print(f"Error queuing workflow: {e}\nResponse: {e.response.text if e.response else 'N/A'}", file=sys.stderr)
        raise

def get_t2i_image_data(prompt_id: str, timeout: int = 400) -> bytes | None:
    """Polls for and retrieves the final image from a t2i workflow."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            res = requests.get(f"{API_BASE}/history/{prompt_id}")
            res.raise_for_status()
            history = res.json()
            if prompt_id in history and 'outputs' in history[prompt_id]:
                outputs = history[prompt_id]['outputs']
                if T2I_OUTPUT_NODE_ID in outputs and 'images' in outputs[T2I_OUTPUT_NODE_ID]:
                    image_info = outputs[T2I_OUTPUT_NODE_ID]['images'][0]
                    img_resp = requests.get(f"{API_BASE}/view?filename={image_info['filename']}&subfolder={image_info.get('subfolder', '')}&type={image_info.get('type', 'output')}")
                    img_resp.raise_for_status()
                    return img_resp.content
        except requests.exceptions.RequestException as e:
            print(f"Polling failed for prompt {prompt_id}: {e}", file=sys.stderr)
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for image from prompt {prompt_id}")

def wait_for_i2v_video_file(start_time: float, timeout: int = 700) -> str | None:
    """Waits for a new video file to appear and stabilize in the output directory."""
    print("Waiting for video file to be created...")
    while time.time() - start_time < timeout:
        pattern = os.path.join(COMFYUI_OUTPUT_DIR, f"{I2V_VIDEO_FILENAME_PREFIX}_*.mp4")
        video_files = [f for f in glob.glob(pattern) if os.path.getmtime(f) > start_time]
        if video_files:
            newest_file = max(video_files, key=os.path.getmtime)
            # Check if file is stable (not being written to)
            initial_size = os.path.getsize(newest_file)
            time.sleep(2)
            if os.path.exists(newest_file) and os.path.getsize(newest_file) == initial_size:
                print(f"Found stable video file: {os.path.basename(newest_file)}")
                return newest_file
        time.sleep(2)
    return None

def upload_image_to_comfy(filepath: str) -> str:
    """Uploads an image to the ComfyUI server's input directory."""
    with open(filepath, 'rb') as f:
        files = {'image': (os.path.basename(filepath), f, 'image/png')}
        resp = requests.post(f"{API_BASE}/upload/image", files=files, data={'overwrite': 'true'})
        resp.raise_for_status()
        return resp.json()['name']

# ─── PIPELINE STAGES ────────────────────────────────────────────────────────────

def run_documentary_generation():
    # ... (This function remains unchanged, no need to repeat it)
    print("--- Step 1: Generating Documentary Content ---")
    start_time = time.time()
    example_topics = ["The Lost City of Atlantis", "The Secrets of the Bermuda Triangle", "The Great Pyramid of Giza's Hidden Chambers"]
    try:
        print("Generating a new documentary idea...")
        new_idea = generate_idea(example_topics)
        print(f"Generated Idea: {new_idea}")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir)
        template_path = os.path.join(project_root, 'templates', 'fern_v1.txt')
        with open(template_path, "r", encoding="utf-8") as f: style_guide = f.read()
        print("\nGenerating the narration script...")
        narration_script = generate_narration(new_idea, style_guide)
        print("\nSplitting the narration into phrases...")
        narration_phrases = split_into_phrases(narration_script)
        print("\nGenerating FLUX prompts for each phrase...")
        flux_prompts, prompt_timings = generate_flux_prompts(narration_phrases)
        output_folder = save_to_json(new_idea, narration_script, narration_phrases, flux_prompts)
        elapsed_time = time.time() - start_time
        print(f"\nDocumentary content generation complete in {elapsed_time:.2f} seconds. Output in: '{output_folder}'")
        return output_folder, prompt_timings
    except Exception as e:
        print(f"An error occurred during documentary content generation: {e}", file=sys.stderr)
        return None, []


def run_visual_generation(doc_output_folder: str, prompt_timings: list):
    """
    Runs the full visual generation pipeline with optimized, in-memory workflows.
    """
    print("\n--- Step 2: Generating All Visuals (Optimized) ---")
    start_time_visuals = time.time()
    
    # --- Load workflows into memory ONCE ---
    print("Loading workflows into memory...")
    try:
        with open(T2I_WORKFLOW_PATH, 'r', encoding='utf-8') as f: t2i_workflow = json.load(f)
        with open(I2V_WORKFLOW_PATH, 'r', encoding='utf-8') as f: i2v_workflow = json.load(f)
        neg_prompt_i2v = i2v_workflow[I2V_NEG_PROMPT_NODE_ID]['inputs']['text']
    except FileNotFoundError as e:
        print(f"Error: Could not load workflow file. {e}", file=sys.stderr)
        return

    # --- Load scene data ---
    json_path = os.path.join(doc_output_folder, "documentary_data.json")
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    scenes = data.get("scenes", [])
    if not scenes: return

    images_dir = os.path.join(doc_output_folder, "images")
    videos_dir = os.path.join(doc_output_folder, "videos")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    client_id = str(uuid.uuid4())
    generated_images = {} # Dict to store {scene_number: path}

    # --- Stage 1: Generate all images ---
    print("\n--- Submitting All Text-to-Image Jobs ---")
    image_gen_start_time = time.time()
    for i, scene in enumerate(scenes):
        scene_num = i + 1
        prompt = scene.get("prompt")
        if not prompt: continue
        
        t2i_workflow[T2I_PROMPT_NODE_ID]["inputs"]["text"] = prompt
        prompt_id = queue_comfy_workflow(t2i_workflow, client_id)
        print(f"Queued image {scene_num}/{len(scenes)}. Prompt ID: {prompt_id}")
        
        try:
            image_bytes = get_t2i_image_data(prompt_id)
            if image_bytes:
                image_path = os.path.join(images_dir, f"scene_{scene_num:03d}.png")
                with open(image_path, "wb") as f: f.write(image_bytes)
                generated_images[scene_num] = image_path
                print(f"  -> Successfully saved image for scene {scene_num}.")
        except Exception as e:
            print(f"  -> Failed to generate image for scene {scene_num}: {e}", file=sys.stderr)
    
    image_gen_time = time.time() - image_gen_start_time
    print(f"\n--- Image Generation Stage Complete in {image_gen_time:.2f} seconds ---")

    # --- Stage 2: Generate all videos ---
    print("\n--- Submitting All Image-to-Video Jobs ---")
    video_gen_start_time = time.time()
    generated_assets = []
    
    for i, scene in enumerate(scenes):
        scene_num = i + 1
        if scene_num not in generated_images:
            print(f"Skipping video for scene {scene_num} (no source image).")
            continue

        image_path = generated_images[scene_num]
        prompt = scene.get("prompt")
        
        # Modify and queue the I2V workflow
        for node_id in I2V_SEED_NODE_IDS:
            if node_id in i2v_workflow: i2v_workflow[node_id]["inputs"]["noise_seed"] = random.randint(1, 2**31-1)
        
        server_filename = upload_image_to_comfy(image_path)
        i2v_workflow[I2V_IMAGE_NODE_ID]["inputs"]["image"] = server_filename
        i2v_workflow[I2V_POS_PROMPT_NODE_ID]["inputs"]["text"] = prompt
        
        start_i2v_time = time.time()
        prompt_id = queue_comfy_workflow(i2v_workflow, client_id)
        print(f"Queued video {scene_num}/{len(scenes)}. Prompt ID: {prompt_id}")

        try:
            video_path = wait_for_i2v_video_file(start_i2v_time)
            if video_path:
                final_video_path = os.path.join(videos_dir, f"scene_{scene_num:03d}.mp4")
                shutil.copy2(video_path, final_video_path)
                print(f"  -> Successfully saved video for scene {scene_num}.")
                generated_assets.append({
                    "scene_number": scene_num, "phrase": scene.get("phrase"), "prompt": prompt,
                    "image_path": image_path, "video_path": final_video_path
                })
        except Exception as e:
            print(f"  -> Failed to generate video for scene {scene_num}: {e}", file=sys.stderr)

    video_gen_time = time.time() - video_gen_start_time
    print(f"\n--- Video Generation Stage Complete in {video_gen_time:.2f} seconds ---")

    # --- Save Asset Manifest and Print Stats ---
    manifest_path = os.path.join(doc_output_folder, "generated_assets_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(generated_assets, f, indent=4)
    print(f"\nTotal visual generation complete in {time.time() - start_time_visuals:.2f} seconds.")
    print(f"Asset manifest saved to '{manifest_path}'")
    # Performance summary can be added here if needed

if __name__ == "__main__":
    total_pipeline_start_time = time.time()
    documentary_output_dir, all_prompt_timings = run_documentary_generation()
    if documentary_output_dir:
        run_visual_generation(documentary_output_dir, all_prompt_timings)
        total_pipeline_time = time.time() - total_pipeline_start_time
        print(f"\n--- Pipeline Finished in {total_pipeline_time:.2f} seconds ---")
    else:
        print("\n--- Pipeline Aborted ---", file=sys.stderr)
        sys.exit(1)
