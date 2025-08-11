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
# Preserve old Deepgram-based narrate as `narrate_deepgram` and switch default to Chatterbox
from narrate import narrate_phrases  # now chatterbox-backed (same interface)
import importlib
try:
    narrate_deepgram = importlib.import_module('narrate_deepgram')
except Exception:
    narrate_deepgram = None
try:
    # Palindrome tripling for videos
    from triple_videos import main as triple_videos_main
except Exception:
    triple_videos_main = None
try:
    # Optional YouTube upload integration
    from upload_youtube import upload_latest as youtube_upload_latest
    from upload_youtube import load_youtube_config as youtube_load_config
except Exception:
    youtube_upload_latest = None
    youtube_load_config = None

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

def set_workflow_image_resolution(workflow: dict, width: int, height: int) -> None:
    """Set width/height on all nodes that expose these inputs (best-effort)."""
    try:
        for node in workflow.values():
            if isinstance(node, dict):
                inputs = node.get("inputs", {})
                if isinstance(inputs, dict):
                    if "width" in inputs:
                        inputs["width"] = width
                    if "height" in inputs:
                        inputs["height"] = height
    except Exception as _:
        # best-effort; if structure differs, skip silently
        pass

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
        idea, theme, visual_style_guide = generate_idea(example_topics, visual_style=os.getenv("VISUAL_STYLE", "photorealistic"))
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
        flux_prompts, prompt_timings = generate_flux_prompts(
            narration_phrases,
            theme,
            visual_style_guide,
            idea=idea,
            visual_style=os.getenv("VISUAL_STYLE", "photorealistic"),
        )
        
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

    # Ensure generated images are HD 1280x720
    set_workflow_image_resolution(t2i_workflow, width=1280, height=720)

    # Style-specific negative prompts (shared + style-differentiated)
    style_key = (os.getenv("VISUAL_STYLE", "photorealistic") or "").strip().lower()
    common_negatives = (
        "text, letters, words, signage, watermark, blurry, motion blur, out of focus, low resolution, pixelated, "
        "noisy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, "
        "fused fingers, duplicate body parts, duplicate hands, duplicate legs, extra limbs, extra fingers, malformed hands, mutated hands, poorly drawn hands, "
        "bad anatomy, weird hand, ugly, silhouette, blacked-out figure, pure black figure, crowd, crowds, many people, large group"
    )
    if style_key in {"low_poly", "low-poly", "low poly", "lowpoly"}:
        style_negatives = (
            ", photorealism, photorealistic, realistic, realistic skin, realistic textures, lifelike faces, detailed pores, "
            "glossy materials, high-frequency textures"
        )
    else:
        # photorealistic default
        style_negatives = (
            ", cartoon, anime, toy, low-poly, low poly, blocky, lego, voxel, stylized, illustration, cel-shaded, comic"
        )
    t2i_negative = common_negatives + style_negatives
    i2v_negative = common_negatives + style_negatives

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
        t2i_workflow["33"]["inputs"]["text"] = t2i_negative
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
        i2v_workflow[I2V_NEG_PROMPT_NODE_ID]["inputs"]["text"] = i2v_negative
        
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

def run_narration_generation(
    doc_output_folder: str,
    narration_phrases: list,
    audio_prompt_path: str = None,
    force_regeneration: bool = False,
):
    """
    Generates narration audio files from the narration script.
    """
    print("\n--- Step 3: Generating Narration Audio ---")
    start_time_narration = time.time()
    try:
        # Default: chatterbox (same filenames/locations). If needed, you can call
        # narrate_deepgram.narrate_phrases(...) elsewhere to use the old engine.
        # Match settings from test_narration.py
        narrate_phrases(
            phrases=narration_phrases,
            output_dir=doc_output_folder,
            audio_prompt_path=audio_prompt_path,
            force_regeneration=force_regeneration,
            exaggeration=0.5,
            temperature=0.7,
            cfg_weight=0.01,
            top_p=1.0,
            min_p=0.05,
            repetition_penalty=1.2,
            end_silence_ms=0,
        )
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

    tripled_dir = os.path.join(output_folder, "tripled_videos")
    videos_dir_fallback = os.path.join(output_folder, "videos")
    videos_dir = tripled_dir if os.path.isdir(tripled_dir) else videos_dir_fallback
    narration_dir = os.path.join(output_folder, "narration")
    temp_dir = os.path.join(output_folder, "temp_assembly")
    final_video_path = os.path.join(output_folder, "final_video_transitions.mp4")
    final_video_with_music_path = os.path.join(output_folder, "final_video_with_music.mp4")

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
            # Add gentle audio fades and a small silence pad to avoid abrupt cuts
            fade_out_dur = 0.15
            fade_in_dur = 0.05
            pad_silence = 0.10
            fade_out_start = max(audio_duration - fade_out_dur, 0.0)
            whole_dur = audio_duration + pad_silence
            audio_filters = (
                f"adelay=1000|1000," # 1-second delay for the narration
                f"afade=t=in:st=0:d={fade_in_dur},"
                f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_dur},"
                f"apad=whole_dur={whole_dur:.3f}"
            )

            if audio_duration > video_duration:
                print(f"  Audio is longer, freezing last frame of video to match audio duration ({audio_duration:.2f}s)")
                ffmpeg_cmd = [
                    "ffmpeg", "-i", video_path, "-i", audio_path,
                    "-vf", f"tpad=stop_mode=clone:stop_duration={audio_duration - video_duration},fps=24,format=yuv420p",
                    "-af", audio_filters,
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", "-y", output_path
                ]
            else:
                print(f"  Video is longer, cutting video to audio duration ({audio_duration:.2f}s)")
                ffmpeg_cmd = [
                    "ffmpeg", "-i", video_path, "-i", audio_path,
                    "-vf", "fps=24,format=yuv420p",
                    "-af", audio_filters,
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

    # Step 4: Mix background music and add fades
    try:
        if not os.path.exists(final_video_path):
            print(f"Error: Concatenated video not found at '{final_video_path}'. Skipping music and fade.", file=sys.stderr)
            return

        # Get video duration for fade-out calculation
        cmd_duration = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", final_video_path]
        video_duration_str = subprocess.check_output(cmd_duration).strip().decode('utf-8')
        video_duration = float(video_duration_str)

        # Add 2 seconds of padding at the end
        video_duration += 2

        project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        music_dir = os.path.join(project_root, "music")
        music_candidates = []
        if os.path.isdir(music_dir):
            for ext in ("*.mp3", "*.wav", "*.webm", "*.m4a", "*.ogg"):
                music_candidates.extend(glob.glob(os.path.join(music_dir, ext)))
        
        fade_duration = 1.5
        fade_out_start = video_duration - fade_duration
        video_fade_filter = f"fade=type=in:duration={fade_duration},fade=type=out:start_time={fade_out_start}:duration={fade_duration}"

        if music_candidates:
            music_track = random.choice(music_candidates)
            print(f"Adding fades and background music: {os.path.basename(music_track)} at 12% volume, starting from 10s.")
            
            # Note: Re-encoding video is necessary for fades. Removed -c:v copy.
            mix_cmd = [
                "ffmpeg",
                "-i", final_video_path,
                "-stream_loop", "-1",
                "-ss", "10",
                "-i", music_track,
                "-filter_complex",
                f"[0:v]{video_fade_filter}[v];"
                f"[1:a]volume=0.12[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=0,"
                f"afade=t=in:d={fade_duration},afade=t=out:st={fade_out_start}:d={fade_duration}[aout]",
                "-map", "[v]",
                "-map", "[aout]",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-shortest",
                "-y", final_video_with_music_path,
            ]
            subprocess.run(mix_cmd, check=True, capture_output=True, text=True)
            print(f"✓ Final video with music and fades saved to: {final_video_with_music_path}")
        else:
            print("No music files found; applying video fades only.")
            fade_only_cmd = [
                "ffmpeg",
                "-i", final_video_path,
                "-vf", video_fade_filter,
                "-af", f"afade=t=in:d={fade_duration},afade=t=out:st={fade_out_start}:d={fade_duration}",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-y", final_video_with_music_path,
            ]
            subprocess.run(fade_only_cmd, check=True, capture_output=True, text=True)
            print(f"✓ Final video with fades saved to: {final_video_with_music_path}")

        # Remove the intermediate file to ensure the version with music/fades is uploaded
        try:
            if os.path.exists(final_video_path):
                print(f"Removing intermediate file: {os.path.basename(final_video_path)}")
                os.remove(final_video_path)
                print("✓ Intermediate file removed.")
        except OSError as e:
            print(f"Warning: Could not remove intermediate file {final_video_path}: {e}", file=sys.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Error while adding fades and/or music: {e.stderr}", file=sys.stderr)
    except Exception as e:
        print(f"✗ Unexpected error while adding fades and/or music: {e}", file=sys.stderr)

    print(f"\n--- Final video assembly completed: {final_video_with_music_path if os.path.exists(final_video_with_music_path) else final_video_path} ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Main pipeline script to generate a complete documentary from an idea.")
    parser.add_argument("--audio_prompt_path", type=str, default="templates/voice_template_smooth.mp3", help="Path to an audio file to use as a voice prompt for narration.")
    parser.add_argument("--regenerate_narration", type=str, help="Path to a project's output folder to regenerate only the narration.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing files.")
    args = parser.parse_args()

    if args.regenerate_narration:
        project_folder = args.regenerate_narration
        json_path = os.path.join(project_folder, "documentary_data.json")
        if not os.path.exists(json_path):
            print(f"Error: Could not find 'documentary_data.json' in '{project_folder}'", file=sys.stderr)
            sys.exit(1)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        narration_phrases = [scene['phrase'] for scene in data.get('scenes', [])]
        
        if not narration_phrases:
            print("No narration phrases found in the project.", file=sys.stderr)
            sys.exit(1)
            
        run_narration_generation(project_folder, narration_phrases, args.audio_prompt_path, force_regeneration=args.force)
        print("\n--- Narration regeneration complete. ---")

    else:
        total_pipeline_start_time = time.time()
        documentary_output_dir, all_prompt_timings, narration_phrases = run_documentary_generation()
        if documentary_output_dir:
            # Run visual and narration generation in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit visual generation to the executor
                visual_future = executor.submit(run_visual_generation, documentary_output_dir, all_prompt_timings)
                
                # Submit narration generation to the executor
                narration_future = executor.submit(run_narration_generation, documentary_output_dir, narration_phrases, args.audio_prompt_path)

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

            # After visuals are generated, create tripled (forward-reverse-forward) clips
            if triple_videos_main is not None:
                try:
                    print("\n--- Step 3.5: Tripling videos with f-r-f effect ---")
                    rc = triple_videos_main(["--output-folder", documentary_output_dir])
                    if rc != 0:
                        print(f"Warning: Tripling videos returned non-zero exit code: {rc}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Tripling videos failed: {e}", file=sys.stderr)

            run_video_assembly(documentary_output_dir)
            total_pipeline_time = time.time() - total_pipeline_start_time
            print(f"\n--- Pipeline Finished in {total_pipeline_time:.2f} seconds ---")

            # Always attempt YouTube upload and write a note on success
            try:
                if youtube_upload_latest is not None:
                    print("\n--- Uploading final video to YouTube ---")
                    video_id = youtube_upload_latest(output_folder=documentary_output_dir)
                    if video_id:
                        note_path = os.path.join(documentary_output_dir, "uploaded_to_youtube.txt")
                        with open(note_path, "w", encoding="utf-8") as f:
                            f.write(f"Uploaded to YouTube. Video ID: {video_id}\n")
                        print(f"✓ Wrote upload note: {note_path}")
                    else:
                        print("YouTube upload did not return a video ID; no note written.", file=sys.stderr)
                else:
                    print("YouTube uploader unavailable; skipping upload.", file=sys.stderr)
            except Exception as e:
                print(f"Warning: YouTube upload step failed: {e}", file=sys.stderr)
        else:
            print("\n--- Pipeline Aborted ---", file=sys.stderr)
            sys.exit(1)
