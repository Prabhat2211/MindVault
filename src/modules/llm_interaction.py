"""
Handles LLM interaction for the Mind Vault Companion using MLX.
"""
import time
from mlx_vlm import generate
from PIL import Image

def get_gemma_response(model, tokenizer, messages, image: Image.Image = None, max_new_tokens=256):
    """
    Generates a response from the Gemma model using MLX.

    Args:
        model: The loaded MLX model.
        tokenizer: The loaded tokenizer.
        messages (list): The conversation history.
        image (PIL.Image, optional): An image to accompany the prompt. Defaults to None.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated text response from the model.
    """
    start_time = time.time()

    if image:
        # For multimodal input, we manually construct the prompt to ensure the <image> token is present.
        # We extract the text from the last user message in the history.
        last_user_message = ""
        if messages and messages[-1]['role'] == 'user':
            last_user_message = messages[-1]['content'][0]['text']
        
        # Manually construct the prompt in the required format for a single-turn image interaction.
        prompt = f"USER: <image>\n{last_user_message}\nASSISTANT:"
    else:
        # For text-only conversations, use the standard method to process history.
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # Generate the response
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        image=image,
        max_tokens=max_new_tokens,
        verbose=False # Set to True for debugging if needed
    )

    end_time = time.time()
    duration = end_time - start_time
    print(f"(Inference time: {duration:.2f} seconds)")

    return response.text