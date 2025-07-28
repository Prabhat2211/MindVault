
"""
Loads the Gemma 3n model for MLX from a local directory.
"""
from pathlib import Path
from mlx_vlm import load

def load_model():
    """
    Loads the MLX-quantized Gemma 3n model and tokenizer from a local directory.

    Returns:
        tuple: A tuple containing:
            - model: The loaded MLX model.
            - tokenizer: The loaded tokenizer.
    """
    model_id = "mlx-community/gemma-3n-E2B-it-4bit"
    # Point to the local directory where the model was downloaded
    model_path = Path(__file__).parent.parent.parent / "models" / model_id

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found at {model_path}. "
            f"Please run the 'scripts/download_model.py' script first."
        )

    print(f"Loading MLX model from local path: {model_path}...")
    
    model, tokenizer = load(str(model_path))
    
    print("MLX model and tokenizer loaded successfully.")
    return model, tokenizer

if __name__ == '__main__':
    try:
        model, tokenizer = load_model()
        print("\nModel and tokenizer loaded successfully for testing.")
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")
    except Exception as e:
        print(f"An error occurred: {e}")
