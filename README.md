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

## Quick TTS with Chatterbox

The script `src/tts_chatterbox.py` generates an MP3 from input text.

Defaults:
- Exaggeration: 0.4
- Temperature: 0.5
- Appends a trailing comma and period by default
- Appends 300 ms of silence at the end

Run without explicitly using the virtual environment (assumes `python` on PATH):

```bash
python src/tts_chatterbox.py "Hello there" --out-dir outputs/tts --filename hello.mp3
```

Optional flags (override defaults):
- `--temperature 0.6`
- `--exaggeration 0.8`
- `--cfg-weight 0.6`
- `--top-p 0.9`
- `--min-p 0.05`
- `--repetition-penalty 1.2`
- `--audio-prompt my_voice.wav`
- `--end-silence-ms 150`
- `--append-comma` / `--append-dot` (both default to on)

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

## Fully Automated UGC Video Generation

The script `src/ugc/upload_to_wangp.py` provides a powerful, end-to-end automation for generating User-Generated Content (UGC) style videos using a local Gradio web application. It intelligently finds the latest creative assets, calculates the precise video length based on audio duration, and drives the web UI to generate a video.

### How It Works

The script automates the entire process from asset collection to final video generation:

1.  **Find Latest Audio File**: Before any browser interaction, the script automatically searches the `outputs/ugc_scripts/Emily_Carter/` directory to find the most recently created subfolder. It then locates the `scene_001.mp3` narration file within that folder. This ensures it always uses the latest audio without any manual path updates.

2.  **Calculate Precise Frame Count**:
    *   It reads the exact duration of the located audio file (e.g., 3.36 seconds).
    *   Using a configurable frame rate (defaulting to 25 FPS), it calculates the ideal number of frames required for the video (e.g., `3.36s * 25fps = 84 frames`).
    *   Crucially, it then adjusts this number to be compatible with the AI model's specific requirements (ensuring `frames - 1` is a multiple of 4), preventing common errors. For 84 frames, it would adjust to 85.

3.  **Drive the Web UI**: The script launches a Playwright-controlled Chromium browser and performs the following actions on the Gradio application running at `http://localhost:7860`:
    *   Uploads a default settings file (`UGC_defaults.json`).
    *   Waits 3 seconds.
    *   Uploads the reference image (`photo_1.png`).
    *   Uploads the `scene_001.mp3` audio file.
    *   Waits 3 seconds.
    *   Inputs the precisely calculated frame count into the correct field.
    *   Waits 1 second.
    *   Clicks the "Generate" button.

4.  **Wait for Completion**: Instead of a fixed wait time, the script intelligently monitors the output directory (`C:\Users\tasos\Code\HunyuanVideo\Wan2GP\outputs`). It checks every 5 seconds for a new `.mp4` file created after the "Generate" button was clicked. This indefinite wait ensures it works for videos of any length.

5.  **Exit Gracefully**: Once a new video is detected, the script prints a success message with the new filename and closes the browser, completing the process.

### Prerequisites

- A running instance of the Gradio/Hunyuan video generation application at `http://localhost:7860`.
- The `mutagen` Python library for reading audio metadata (`pip install mutagen`).
- The necessary output directory structure must exist for the script to monitor (`C:\Users\tasos\Code\HunyuanVideo\Wan2GP\outputs`).

### How to Run

Ensure all prerequisite services are running and file paths at the top of the script are correct. Then, execute the script from the project root:

```bash
# Ensure your virtual environment is activated
venv\Scripts\python.exe src/ugc/upload_to_wangp.py
```

The script will provide detailed print statements in the console for every step it takes, from file analysis to browser automation and final video detection.
