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

# Import the main functions from the UGC scripts.
from src.ugc._01_script_generation import run_generate_ugc_script
from src.ugc._02_edit_image import run_image_editing_for_pipeline
from src.ugc._03_script_narration import run_ugc_narration
from src.ugc._04_wangp_product import run_wangp_product_generation
from src.ugc._05_wangp_character import run_wangp_character_generation


def main():
    """
    Executes the UGC generation, image editing, narration, and video generation pipeline.
    """
    print("--- Starting UGC Pipeline ---")
    
    # --- Step 1: Generate the UGC script ---
    print("\n[Step 1/5] Generating UGC script...")
    script_result = run_generate_ugc_script()
    
    # If script generation fails, the function will return None.
    # In that case, we stop the pipeline here.
    if not script_result:
        print("\nPipeline stopped: Script generation failed.")
        sys.exit(1)
    
    generated_script_path, product_image_filename = script_result
    print(f"Script generation successful. Output at: {generated_script_path}")
    
    # --- Step 2: Edit product image ---
    print("\n[Step 2/5] Editing product image...")
    
    # Use the same directory as the script for output
    script_dir = os.path.dirname(generated_script_path)
    
    # Call the image editing function with the product filename and script directory
    success = run_image_editing_for_pipeline(product_image_filename, script_dir)
    
    if success:
        print("Image editing completed successfully.")
    else:
        print("Image editing failed.")
    
    # --- Step 3: Narrate the generated script ---
    print("\n[Step 3/5] Narrating generated script...")
    run_ugc_narration(generated_script_path)
    
    # --- Step 4: Generate videos with WanGP ---
    print("\n[Step 4/5] Generating Product Videos with WanGP...")
    wangp_success = run_wangp_product_generation(script_dir)
    
    if wangp_success:
        print("WanGP product video generation completed successfully.")
    else:
        print("WanGP product video generation failed.")
        
    # --- Step 5: Generate Character Videos ---
    print("\n[Step 5/5] Generating Character Videos with WanGP...")
    character_success = run_wangp_character_generation(script_dir)

    if character_success:
        print("WanGP character video generation completed successfully.")
    else:
        print("WanGP character video generation failed.")
    
    print("\n--- UGC Pipeline Finished ---")


if __name__ == "__main__":
    main()

