import requests
import json
import sys

API_BASE = "http://127.0.0.1:8188"

def check_prompt_status(prompt_id: str):
    """
    Fetches and prints the history for a specific prompt_id from the ComfyUI API.
    """
    if not prompt_id:
        print("Error: Please provide a prompt_id.", file=sys.stderr)
        print("Usage: python src/check_prompt.py <prompt_id>", file=sys.stderr)
        return

    try:
        print(f"Fetching history for prompt_id: {prompt_id}")
        response = requests.get(f"{API_BASE}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json()
        print("\n--- Full History Response ---")
        print(json.dumps(history, indent=4))
        
        if prompt_id in history:
            print(f"\n--- Details for Prompt {prompt_id} ---")
            print(json.dumps(history[prompt_id], indent=4))
        else:
            print(f"\nPrompt {prompt_id} not found in the history response.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ComfyUI API: {e}", file=sys.stderr)
        if e.response:
            print(f"Response Status: {e.response.status_code}", file=sys.stderr)
            print(f"Response Body: {e.response.text}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt_id_to_check = sys.argv[1]
        check_prompt_status(prompt_id_to_check)
    else:
        print("Usage: python src/check_prompt.py <prompt_id>", file=sys.stderr)
