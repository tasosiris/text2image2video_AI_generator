# Automated Documentary Video Generator

This project is a fully automated pipeline that generates short documentary-style videos from a single topic idea. It leverages generative AI for every step of the process, from creating the narrative and script to generating the visual scenes and compiling them into video clips.

## Overview

The main pipeline, orchestrated by `src/run_pipeline.py`, performs the following steps:

1.  **Content Generation**: 
    *   Takes a list of example topics and generates a new, unique documentary idea.
    *   Writes a full narration script based on the generated idea and a stylistic guide.
    *   Splits the narration into individual phrases, which will become scenes in the video.
    *   Generates descriptive visual prompts (for an image model) for each phrase.

2.  **Visual Generation**:
    *   Connects to a running **ComfyUI** instance via its API.
    *   **Text-to-Image**: Submits all the generated prompts to a text-to-image workflow to create the base images for each scene.
    *   **Image-to-Video**: Takes the generated images and submits them to an image-to-video workflow to create short video clips for each scene.
    *   The generation is done concurrently for efficiency.

3.  **Output**:
    *   Saves all generated content—idea, narration, prompts, images, and videos—into a well-organized folder structure within the `outputs` directory.
    *   Creates a `generated_assets_manifest.json` file to easily track all the assets for a given documentary.

## Prerequisites

- Python 3.8+
- A running instance of [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- The specific ComfyUI custom nodes and models required by the workflows you intend to use.

## Setup & Configuration

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Paths in `run_pipeline.py`:**

    Before running the script, you **must** update the hardcoded paths and node IDs at the top of `src/run_pipeline.py`. These need to point to your local ComfyUI setup and the specific workflows you are using.

    ```python
    # --- ComfyUI API Configuration ---
    API_BASE = "http://127.0.0.1:8188"
    T2I_WORKFLOW_PATH = r"C:\path\to\your\text_to_image_workflow.json"
    I2V_WORKFLOW_PATH = r"C:\path\to\your\image_to_video_workflow.json"
    COMFYUI_OUTPUT_DIR = r"C:\path\to\your\ComfyUI\output"

    # --- Node IDs for Workflows ---
    # (Update these to match the node IDs in your specific JSON workflows)
    T2I_PROMPT_NODE_ID = "6" 
    T2I_OUTPUT_NODE_ID = "9"
    I2V_IMAGE_NODE_ID = "1206"
    # ... and so on for other node IDs.
    ```
    To get the workflow `.json` files, open your workflow in ComfyUI and click **"Save (API Format)"**.

## How to Run

Once the configuration is complete, you can start the entire documentary generation process by running the main pipeline script:

```bash
python src/run_pipeline.py
```

The script will print its progress to the console, and you can find the final output in a newly created folder inside the `/outputs` directory.

## YouTube Upload (Optional)

Add one-click upload of the latest final video to YouTube.

1. Enable the YouTube Data API v3 in your Google Cloud project and create an OAuth 2.0 Client ID (Desktop App).
2. Save the downloaded client secret JSON as `config/client_secret.json`.
3. Copy the example config and adjust as needed:
   ```bash
   cp config/youtube_config.example.json config/youtube_config.json
   ```
4. Install uploader dependencies:
   ```bash
   pip install -r requirements.youtube.txt
   ```
5. First-time auth (opens browser):
   ```bash
   python src/upload_youtube.py --auth
   ```
6. Upload latest final video:
   ```bash
   python src/upload_youtube.py
   ```

To auto-upload at the end of the pipeline, set `"auto_upload": true` in `config/youtube_config.json`.

## Project Structure

```
.
├── config/             # Stores workflow configurations.
├── outputs/            # Default directory for all generated documentaries (ignored by git).
├── src/                # Main source code for the pipeline.
│   ├── documentary_generator.py # Handles all text-based generation.
│   └── run_pipeline.py # The main entry point to run the full pipeline.
├── templates/          # Contains style guides for narration generation.
├── .gitignore          # Specifies files and directories to be ignored by git.
├── README.md           # This file.
└── requirements.txt    # Python dependencies.
```
