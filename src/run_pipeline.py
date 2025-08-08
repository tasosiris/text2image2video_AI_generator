#!/usr/bin/env python3
"""
Main pipeline script to generate a complete documentary from an idea.
This script orchestrates the entire content generation process in stages for efficiency:
1.  **Content Generation:** Creates documentary idea, theme, visual style guide, narration, and prompts.
2.  **Visual Generation:**
    a. Loads ComfyUI workflows into memory once.
    b. Submits all text-to-image jobs.
    c. Submits all image-to-video jobs.
3.  **Narration Generation:** Creates audio files for each narration phrase.
4.  **Asset Manifest & Assembly:** Saves a manifest of all assets and assembles the final video.

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
import subprocess
from documentary_generator import (
    generate_idea, generate_narration, split_into_phrases,
    generate_flux_prompts, save_to_json
)
from narrate import narrate_phrases

# --- ComfyUI API Configuration ---
API_BASE = "http://127.0.0.1:8188"
T2I_WORKFLOW_PATH = r"C:\ComfyUI\ComfyUI_windows_portable\ComfyUI\workflows\api+workflows\flux_krea_faster.json"
I2V_WORKFLOW_PATH = r"C:\ComfyUI\ComfyUI_windows_portable\ComfyUI\workflows\api+workflows\LTXV_13B_Upscale.json"
COMFYUI_OUTPUT_DIR = r"C:\ComfyUISage\ComfyUI-Easy-Install\ComfyUI\output"

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
    """Generates the core content: idea, theme, style guide, narration, and prompts."""
    print("--- Step 1: Generating Documentary Content ---")
    start_time = time.time()
    example_topics = [
        "The Battle of Salamis: How a Navy Saved Greece",
        "Rome's Concrete Revolution: The Secret of Pozzolana",
        "Carthage vs. Rome: Logistics that Won the Punic Wars"
    ]
    try:
        print("Generating a new documentary idea, theme, and style guide...")
        idea, theme, visual_style_guide = generate_idea(example_topics)
        print(f"Generated Idea: {idea}")
        print(f"Theme: {theme}")
        print(f"Visual Style Guide: {visual_style_guide}")

        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir)
        template_path = os.path.join(project_root, 'templates', 'fern_v1.txt')
        with open(template_path, "r", encoding="utf-8") as f: style_guide = f.read()
        
        print("\nGenerating the narration script...")
        narration_script = generate_narration(idea, style_guide)
        
        print("\nSplitting the narration into phrases...")
        narration_phrases = split_into_phrases(narration_script)
        
        print("\nGenerating FLUX prompts for each phrase...")
        flux_prompts, prompt_timings = generate_flux_prompts(narration_phrases, theme, visual_style_guide)
        
        output_folder = save_to_json(idea, theme, visual_style_guide, narration_script, narration_phrases, flux_prompts)
        
        elapsed_time = time.time() - start_time
        print(f"\nDocumentary content generation complete in {elapsed_time:.2f} seconds. Output in: '{output_folder}'")
        
        return output_folder, prompt_timings, narration_phrases
    except Exception as e:
        print(f"An error occurred during documentary content generation: {e}", file=sys.stderr)
        return None, [], []


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
        t2i_workflow["33"]["inputs"]["text"] = "lego, toy, blocky, minecraft style, boxxy figures, blurry, motion blur, out of focus, low resolution, pixelated, noisy, text, letters, words, signage, watermark, ugly, deformed, disfigured, bad anatomy, duplicate body parts, extra limbs, extra legs, duplicate hands, duplicate legs, extra fingers, fused fingers, malformed hands, mutated hands, poorly drawn hands, Photorealism, cluttered backgrounds, facial detail, glossy materials, high-frequency textures, decals, crowds, soft shading, smooth organic curves, cartoon exaggeration, busy compositions, LEGO-like shapes, reflective surfaces, bright modern overlays, excessive detail in small objects, complex patterns, low quality, worst quality, distorted, motion smear, motion artifacts, weird hand."
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
        
        for node_id in I2V_SEED_NODE_IDS:
            if node_id in i2v_workflow: i2v_workflow[node_id]["inputs"]["noise_seed"] = random.randint(1, 2**31-1)
        
        server_filename = upload_image_to_comfy(image_path)
        i2v_workflow[I2V_IMAGE_NODE_ID]["inputs"]["image"] = server_filename
        i2v_workflow[I2V_POS_PROMPT_NODE_ID]["inputs"]["text"] = prompt
        i2v_workflow[I2V_NEG_PROMPT_NODE_ID]["inputs"]["text"] = "text, letters, words, signage, watermark, blurry, motion blur, out of focus, low resolution, pixelated, noisy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
        
        start_i2v_time = time.time()
        prompt_id = queue_comfy_workflow(i2v_workflow, client_id)
        print(f"Queued video {scene_num}/{len(scenes)}. Prompt ID: {prompt_id}")

        try:
            video_path = wait_for_i2v_video_file(start_i2v_time)
            if video_path:
                time.sleep(1)  # Wait an extra second to ensure the file is fully written
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

    manifest_path = os.path.join(doc_output_folder, "generated_assets_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(generated_assets, f, indent=4)
    print(f"\nTotal visual generation complete in {time.time() - start_time_visuals:.2f} seconds.")
    print(f"Asset manifest saved to '{manifest_path}'")

def run_narration_generation(doc_output_folder: str, narration_phrases: list):
    """
    Generates narration audio files from the narration script.
    """
    print("\n--- Step 3: Generating Narration Audio ---")
    start_time_narration = time.time()
    try:
        narrate_phrases(narration_phrases, doc_output_folder)
        elapsed_time = time.time() - start_time_narration
        print(f"\nNarration audio generation complete in {elapsed_time:.2f} seconds.")
    except Exception as e:
        print(f"An error occurred during narration generation: {e}", file=sys.stderr)

def run_video_assembly(output_folder):
    """
    Assembles the final documentary video using direct FFmpeg commands for robust,
    artifact-free audio synchronization and adds cross-fade transitions.
    """
    print(f"\n--- Step 4: Assembling Final Video for: {os.path.basename(output_folder)} ---")

    videos_dir = os.path.join(output_folder, "videos")
    narration_dir = os.path.join(output_folder, "narration")
    temp_dir = os.path.join(output_folder, "temp_assembly")
    final_video_path = os.path.join(output_folder, "final_video_transitions.mp4")

    os.makedirs(temp_dir, exist_ok=True)

    video_files = sorted(glob.glob(os.path.join(videos_dir, "scene_*.mp4")))
    narration_files = sorted(glob.glob(os.path.join(narration_dir, "narration_*.mp3")))

    if not video_files or not narration_files:
        print("Error: No video or narration files found.", file=sys.stderr)
        return

    combined_clips = []
    print("Step 1: Combining video and audio pairs...")

    for i, (video_path, audio_path) in enumerate(zip(video_files, narration_files)):
        scene_num = i + 1
        output_path = os.path.join(temp_dir, f"combined_scene_{scene_num:03d}.ts")
        
        try:
            cmd_video_duration = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            video_duration = float(subprocess.check_output(cmd_video_duration).strip())

            cmd_audio_duration = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            audio_duration = float(subprocess.check_output(cmd_audio_duration).strip())
            
            ffmpeg_cmd = []
            if audio_duration > video_duration:
                print(f"  Audio is longer, freezing last frame of video to match audio duration ({audio_duration:.2f}s)")
                ffmpeg_cmd = [
                    "ffmpeg", "-i", video_path, "-i", audio_path,
                    "-vf", f"tpad=stop_mode=clone:stop_duration={audio_duration - video_duration},fps=24,format=yuv420p",
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", "-y", output_path
                ]
            else:
                print(f"  Video is longer, cutting video to audio duration ({audio_duration:.2f}s)")
                ffmpeg_cmd = [
                    "ffmpeg", "-i", video_path, "-i", audio_path,
                    "-vf", "fps=24,format=yuv420p",
                    "-t", str(audio_duration),
                    "-c:v", "libx264", "-c:a", "aac", "-y", output_path
                ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"✓ Combined scene {scene_num} saved to: {os.path.basename(output_path)}")
            combined_clips.append(output_path)

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"✗ Error processing scene {scene_num}: {e}", file=sys.stderr)
            if isinstance(e, subprocess.CalledProcessError):
                print(f"FFmpeg stderr: {e.stderr}", file=sys.stderr)
            continue

    if not combined_clips:
        print("No clips were processed successfully. Aborting.", file=sys.stderr)
        return

    print(f"\nStep 2: Concatenating {len(combined_clips)} clips...")
    
    file_list_path = os.path.join(temp_dir, "file_list.txt")
    with open(file_list_path, "w") as f:
        for clip_path in combined_clips:
            f.write(f"file '{os.path.abspath(clip_path)}'\n")

    concat_cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", file_list_path,
        "-c", "copy", "-y", final_video_path
    ]
    
    try:
        subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
        print(f"✓ Final video saved to: {final_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during final concatenation: {e.stderr}", file=sys.stderr)

    print("\nStep 3: Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("✓ Cleanup completed")
    print(f"\n--- Final video assembly completed: {final_video_path} ---")


if __name__ == "__main__":
    total_pipeline_start_time = time.time()
    documentary_output_dir, all_prompt_timings, narration_phrases = run_documentary_generation()
    if documentary_output_dir:
        # Run visual and narration generation in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit visual generation to the executor
            visual_future = executor.submit(run_visual_generation, documentary_output_dir, all_prompt_timings)
            
            # Submit narration generation to the executor
            narration_future = executor.submit(run_narration_generation, documentary_output_dir, narration_phrases)

            # Wait for both tasks to complete
            concurrent.futures.wait([visual_future, narration_future])

            # Optionally, check for exceptions
            try:
                visual_future.result()
            except Exception as e:
                print(f"Visual generation failed: {e}", file=sys.stderr)
            
            try:
                narration_future.result()
            except Exception as e:
                print(f"Narration generation failed: {e}", file=sys.stderr)

        run_video_assembly(documentary_output_dir)
        total_pipeline_time = time.time() - total_pipeline_start_time
        print(f"\n--- Pipeline Finished in {total_pipeline_time:.2f} seconds ---")
    else:
        print("\n--- Pipeline Aborted ---", file=sys.stderr)
        sys.exit(1)
