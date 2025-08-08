#!/usr/bin/env python3
"""
Standalone script to assemble the final video from an existing output folder.

This script finds the most recent documentary output folder and calls the 
video assembly logic from the main pipeline script to generate the final video.
"""
import os
import sys

# Add the 'src' directory to the Python path to allow for module imports
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)

try:
    from run_pipeline import run_video_assembly
except ImportError as e:
    print(f"Error: Could not import assembly logic from 'run_pipeline.py'.\n{e}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    project_root = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_root, 'outputs')

    if not os.path.isdir(outputs_dir):
        print(f"Error: The 'outputs' directory was not found at '{outputs_dir}'", file=sys.stderr)
        sys.exit(1)

    # Find all subdirectories in the outputs folder
    all_output_folders = [
        os.path.join(outputs_dir, d)
        for d in os.listdir(outputs_dir)
        if os.path.isdir(os.path.join(outputs_dir, d))
    ]

    if not all_output_folders:
        print("No output folders found in the 'outputs' directory.", file=sys.stderr)
        sys.exit(1)

    # Find the most recently modified folder
    try:
        latest_folder = max(all_output_folders, key=os.path.getmtime)
    except (ValueError, OSError) as e:
        print(f"Error finding the latest folder: {e}", file=sys.stderr)
        sys.exit(1)

    if latest_folder:
        print(f"--- Found latest output folder to process: {os.path.basename(latest_folder)} ---")
        # Call the assembly function from the main pipeline
        run_video_assembly(latest_folder)
    else:
        print("Could not determine the latest output folder.", file=sys.stderr)
        sys.exit(1)
