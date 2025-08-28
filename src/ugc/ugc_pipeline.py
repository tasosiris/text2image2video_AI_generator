"""
This script runs the full UGC content creation pipeline, from script generation
to voice narration, by calling the other UGC scripts in sequence.
"""

import sys
import os

# Adjust the Python path to allow for imports from the 'src' directory.
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

# Import the main functions from the other two UGC scripts.
from src.ugc.generate_ugc_script import run_generate_ugc_script
from src.ugc.ugc_narration import run_ugc_narration


def main():
    """
    Executes the UGC generation and narration pipeline.
    """
    print("--- Starting UGC Pipeline ---")
    
    # --- Step 1: Generate the UGC script ---
    print("\n[Step 1/2] Generating UGC script...")
    generated_script_path = run_generate_ugc_script()
    
    # If script generation fails, the function will return None.
    # In that case, we stop the pipeline here.
    if not generated_script_path:
        print("\nPipeline stopped: Script generation failed.")
        sys.exit(1)
        
    print(f"Script generation successful. Output at: {generated_script_path}")
    
    # --- Step 2: Narrate the generated script ---
    print("\n[Step 2/2] Narrating generated script...")
    run_ugc_narration(generated_script_path)
    
    print("\n--- UGC Pipeline Finished ---")


if __name__ == "__main__":
    main()

