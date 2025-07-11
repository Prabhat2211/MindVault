import os
import torch
from transformers import AutoProcessor, AutoTokenizer, Gemma3nForConditionalGeneration
from dotenv import load_dotenv

def load_model():
    """
    Loads the Gemma 3n model, processor, and tokenizer.

    This function loads the specified Gemma 3n model and its associated
    processor and tokenizer from Hugging Face. It automatically detects
    if a CUDA-enabled GPU is available and loads the model onto it,
    otherwise it uses the CPU. It retrieves the Hugging Face Hub token
    from a .env file.

    Returns:
        tuple: A tuple containing:
            - model (Gemma3nForConditionalGeneration): The loaded model.
            - processor (AutoProcessor): The loaded processor.
            - tokenizer (AutoTokenizer): The loaded tokenizer.
    """
    load_dotenv()
    
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set HUGGING_FACE_HUB_TOKEN in your .env file.")

    # 1. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define model ID and data type
    model_id = "google/gemma-3n-E2B-it"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # 3. Load tokenizer, processor, and model
    print("Loading model components...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    print("Model components loaded successfully.")

    return model, processor, tokenizer

if __name__ == '__main__':
    # Example of how to use the function
    try:
        model, processor, tokenizer = load_model()
        print("\nModel, processor, and tokenizer loaded successfully for testing.")
        print(f"Model class: {model.__class__.__name__}")
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    except Exception as e:
        print(f"An error occurred: {e}")
