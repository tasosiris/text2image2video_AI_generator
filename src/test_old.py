import torch
from diffusers import FluxPipeline
from dotenv import load_dotenv
import os

# --- Setup ---
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_TOKEN")

# --- Load Model (once) ---
print("Loading model... This will only happen once.")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16,
    token=token
)
print("Model loaded. Ready for prompts!")

# --- Generation Loop ---
image_count = 1
while True:
    prompt = input("\nEnter a prompt (or press Enter to exit): ")
    
    if not prompt:
        print("Exiting.")
        break
        
    print(f"Generating image for prompt: \"{prompt}\"")
    
    # The generator is not set, so you will get a different image each time
    # for the same prompt.
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
    ).images[0]
    
    # Save with a unique name
    filename = f"output_{image_count}.png"
    image.save(filename)
    print(f"Image saved as {filename}")
    image_count += 1
