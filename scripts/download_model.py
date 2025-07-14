
"""
Downloads a model from the Hugging Face Hub to a local directory.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

def download_model():
    """
    Downloads the specified model from Hugging Face Hub.
    """
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("Hugging Face token not found in .env file, proceeding without it.")
        print("If the model is private, this will fail.")

    model_id = "mlx-community/gemma-3n-E2B-it-lm-4bit"
    # Create a path within the project's models directory
    local_dir = Path(__file__).parent.parent / "models" / model_id

    print(f"Downloading model: {model_id} to {local_dir}")

    # Ensure the target directory exists
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # Set to False to download actual files
            token=hf_token
        )
        print("\nModel downloaded successfully!")
    except Exception as e:
        print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    download_model()

