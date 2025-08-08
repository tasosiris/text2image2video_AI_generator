#!/usr/bin/env python3
"""
Test pipeline script to generate a small, 3-scene documentary with performance timing.

This script runs a miniature version of the main pipeline to quickly test all stages:
1.  **Content Generation:** Creates idea and a full script, but truncates to 3 sentences.
2.  **Visual Generation:** Generates 3 images and 3 videos concurrently.
3.  **Narration Generation:** Generates 3 audio files concurrently.
4.  **Performance Summary:** Prints timings for each stage and calculates average per-asset times.
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
from narrate import narrate_from_file, split_text_into_sentences

# --- ComfyUI API Configuration (Same as main pipeline) ---
API_BASE = "http://127.0.0.1:8188"
T2I_WORKFLOW_PATH = r"C:\ComfyUI\ComfyUI_windows_portable\ComfyUI\workflows\api+workflows\workflow-comfyui-workflow-for-flux-simple-iuRdGnfzmTbOOzONIiVV-maitruclam-openart.ai.json"
I2V_WORKFLOW_PATH = r"C:\ComfyUI\ComfyUI_windows_portable\ComfyUI\workflows\api+workflows\LTXV_workflow_optimized.json"
COMFYUI_OUTPUT_DIR = r"C:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output"

# Node IDs
T2I_PROMPT_NODE_ID = "6"
T2I_OUTPUT_NODE_ID = "9"
I2V_IMAGE_NODE_ID = "1206"
I2V_POS_PROMPT_NODE_ID = "6"
I2V_NEG_PROMPT_NODE_ID = "7"
I2V_SEED_NODE_IDS = ["1507", "1598"]
I2V_VIDEO_FILENAME_PREFIX = "ltxv-up"

# --- ComfyUI API Helpers (copied from main pipeline) ---

def queue_comfy_workflow(workflow: dict, client_id: str) -> str:
    """Queues a workflow on the ComfyUI server."""
    try:
        payload = {"prompt": workflow, "client_id": client_id}
        resp = requests.post(f"{API_BASE}/prompt", json=payload)
        resp.raise_for_status()
        return resp.json()["prompt_id"]
    except requests.exceptions.RequestException as e:
        print(f"Error queuing workflow: {e}", file=sys.stderr)
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
        except requests.exceptions.RequestException:
            time.sleep(1) # Ignore and retry
    raise TimeoutError(f"Timed out waiting for image from prompt {prompt_id}")

def wait_for_i2v_video_file(start_time: float, timeout: int = 700) -> str | None:
    """Waits for a new video file to appear and stabilize in the output directory."""
    while time.time() - start_time < timeout:
        pattern = os.path.join(COMFYUI_OUTPUT_DIR, f"{I2V_VIDEO_FILENAME_PREFIX}_*.mp4")
        video_files = [f for f in glob.glob(pattern) if os.path.getmtime(f) > start_time]
        if video_files:
            newest_file = max(video_files, key=os.path.getmtime)
            time.sleep(2) # Wait for file to finish writing
            if os.path.exists(newest_file):
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

# --- PIPELINE STAGES FOR TESTING ---

def run_test_content_generation(limit=3):
    """
    Generates documentary content but truncates it to a specific number of sentences.
    """
    print(f"--- Step 1: Generating Test Content ({limit} Scenes) ---")
    start_time = time.time()
    try:
        # Generate the full content
        new_idea = generate_idea(["Test Topic"])
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir)
        template_path = os.path.join(project_root, 'templates', 'fern_v1.txt')
        with open(template_path, "r", encoding="utf-8") as f: style_guide = f.read()
        
        narration_script = generate_narration(new_idea, style_guide)
        
        # Truncate to the first `limit` sentences
        sentences = split_text_into_sentences(narration_script)
        limited_narration = " ".join(sentences[:limit])
        narration_phrases = split_into_phrases(limited_narration)
        
        flux_prompts, _ = generate_flux_prompts(narration_phrases)

        # Ensure we only have `limit` items
        narration_phrases = narration_phrases[:limit]
        flux_prompts = flux_prompts[:limit]

        output_folder = save_to_json(new_idea, limited_narration, narration_phrases, flux_prompts)
        
        elapsed_time = time.time() - start_time
        print(f"Content generation complete in {elapsed_time:.2f} seconds. Output: '{output_folder}'")
        return output_folder
    except Exception as e:
        print(f"Error during content generation: {e}", file=sys.stderr)
        return None

def run_test_visual_generation(doc_output_folder: str):
    """
    Runs a timed visual generation pipeline for the limited scenes.
    """
    print("\n--- Step 2: Generating Test Visuals ---")
    
    # --- Load workflows ---
    try:
        with open(T2I_WORKFLOW_PATH, 'r') as f: t2i_workflow = json.load(f)
        with open(I2V_WORKFLOW_PATH, 'r') as f: i2v_workflow = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading workflow: {e}", file=sys.stderr)
        return
        
    # --- Load scene data ---
    json_path = os.path.join(doc_output_folder, "documentary_data.json")
    with open(json_path, 'r') as f: data = json.load(f)
    scenes = data.get("scenes", [])
    if not scenes: return

    images_dir = os.path.join(doc_output_folder, "images")
    videos_dir = os.path.join(doc_output_folder, "videos")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    client_id = str(uuid.uuid4())
    image_times = []
    video_times = []

    # --- Generate Images ---
    print(f"\nGenerating {len(scenes)} images...")
    total_image_start_time = time.time()
    generated_images = {}
    for i, scene in enumerate(scenes):
        scene_num = i + 1
        prompt = scene.get("prompt")
        t2i_workflow[T2I_PROMPT_NODE_ID]["inputs"]["text"] = prompt
        
        img_start = time.time()
        prompt_id = queue_comfy_workflow(t2i_workflow, client_id)
        try:
            image_bytes = get_t2i_image_data(prompt_id)
            if image_bytes:
                image_path = os.path.join(images_dir, f"scene_{scene_num:03d}.png")
                with open(image_path, "wb") as f: f.write(image_bytes)
                generated_images[scene_num] = image_path
                img_time = time.time() - img_start
                image_times.append(img_time)
                print(f"  - Image {scene_num} generated in {img_time:.2f}s")
        except Exception as e:
            print(f"  - Failed to generate image for scene {scene_num}: {e}", file=sys.stderr)
    total_image_time = time.time() - total_image_start_time
    
    # --- Generate Videos ---
    print(f"\nGenerating {len(generated_images)} videos...")
    total_video_start_time = time.time()
    for i, scene in enumerate(scenes):
        scene_num = i + 1
        if scene_num not in generated_images: continue
        
        image_path = generated_images[scene_num]
        prompt = scene.get("prompt")
        
        vid_start = time.time()
        server_filename = upload_image_to_comfy(image_path)
        i2v_workflow[I2V_IMAGE_NODE_ID]["inputs"]["image"] = server_filename
        i2v_workflow[I2V_POS_PROMPT_NODE_ID]["inputs"]["text"] = prompt
        
        prompt_id = queue_comfy_workflow(i2v_workflow, client_id)
        try:
            video_path = wait_for_i2v_video_file(vid_start)
            if video_path:
                final_video_path = os.path.join(videos_dir, f"scene_{scene_num:03d}.mp4")
                shutil.copy2(video_path, final_video_path)
                vid_time = time.time() - vid_start
                video_times.append(vid_time)
                print(f"  - Video {scene_num} generated in {vid_time:.2f}s")
        except Exception as e:
            print(f"  - Failed to generate video for scene {scene_num}: {e}", file=sys.stderr)
    total_video_time = time.time() - total_video_start_time
    
    return total_image_time, image_times, total_video_time, video_times

def run_test_narration_generation(doc_output_folder: str):
    """
    Runs a timed narration generation pipeline.
    """
    print("\n--- Step 3: Generating Test Narration Audio ---")
    narration_start_time = time.time()
    try:
        narrate_from_file(doc_output_folder)
        narration_time = time.time() - narration_start_time
        return narration_time
    except Exception as e:
        print(f"Error during narration generation: {e}", file=sys.stderr)
        return 0

# --- MAIN TEST EXECUTION ---

if __name__ == "__main__":
    total_start_time = time.time()
    
    # 1. Content
    output_dir = run_test_content_generation(limit=3)
    
    if output_dir:
        # 2. Visuals
        img_total, img_times, vid_total, vid_times = run_test_visual_generation(output_dir)
        
        # 3. Narration
        narration_total = run_test_narration_generation(output_dir)
        
        # 4. Performance Summary
        total_time = time.time() - total_start_time
        print("\n--- TEST PIPELINE PERFORMANCE SUMMARY ---")
        print(f"  - Total Pipeline Time: {total_time:.2f} seconds")
        print("-" * 40)
        
        # Image Stats
        if img_times:
            avg_img = sum(img_times) / len(img_times)
            print(f"  - Image Generation ({len(img_times)} files):")
            print(f"    - Total Time: {img_total:.2f}s")
            print(f"    - Average Time per Image: {avg_img:.2f}s")
        
        # Video Stats
        if vid_times:
            avg_vid = sum(vid_times) / len(vid_times)
            print(f"  - Video Generation ({len(vid_times)} files):")
            print(f"    - Total Time: {vid_total:.2f}s")
            print(f"    - Average Time per Video: {avg_vid:.2f}s")

        # Narration Stats
        # Since narration is concurrent, we measure total time and divide by number of files
        num_narration_files = len(glob.glob(os.path.join(output_dir, "narration", "*.mp3")))
        if num_narration_files > 0:
            avg_narr = narration_total / num_narration_files
            print(f"  - Narration Generation ({num_narration_files} files):")
            print(f"    - Total Time: {narration_total:.2f}s (concurrent)")
            print(f"    - Average Time per File: {avg_narr:.2f}s")

    else:
        print("\n--- Test Pipeline Aborted ---", file=sys.stderr)
        sys.exit(1)
