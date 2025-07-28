
"""
Handles LLM interaction for the Mind Vault Companion using MLX.
"""
import time
from mlx_vlm import generate

def get_gemma_response(model, tokenizer, messages, max_new_tokens=256):
    """
    Generates a response from the Gemma model using MLX.

    Args:
        model: The loaded MLX model.
        tokenizer: The loaded tokenizer.
        messages (list): The conversation history.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated text response from the model.
    """
    start_time = time.time()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        verbose=False
    )

    end_time = time.time()
    duration = end_time - start_time
    print(f"(Inference time: {duration:.2f} seconds)")

    # The generate function returns a GenerationResult object, so we extract the text.
    return response.text

if __name__ == '__main__':
    # This is a placeholder for a test.
    # To run a full test, we would need to load the model, which is slow.
    print("Testing llm_interaction module with dummy MLX components...")

    class DummyModel:
        pass

    class DummyTokenizer:
        def apply_chat_template(self, conv, tokenize, add_generation_prompt):
            # Combine messages into a single string for the dummy prompt
            full_prompt = ""
            for msg in conv:
                full_prompt += f"{msg['role']}: {msg['content'][0]['text']}\n"
            return full_prompt

    # Mimic the GenerationResult object returned by mlx_vlm.generate
    class DummyGenerationResult:
        def __init__(self, text):
            self.text = text

    def dummy_generate(model, tokenizer, prompt, max_tokens, verbose):
        return DummyGenerationResult("This is a dummy response from the MLX model.")

    # Replace the actual generate function with the dummy one for the test
    import sys
    # In a real scenario, you wouldn't do this, but it's for a simple standalone test.
    # We are replacing the function in the current module's scope for the test.
    original_generate = generate
    sys.modules[__name__].generate = dummy_generate

    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello, how are you?"}]
        }
    ]

    print(f"\nGenerating response for: {messages[0]['content'][0]['text']}")

    response = get_gemma_response(
        dummy_model,
        dummy_tokenizer,
        messages,
        max_new_tokens=50
    )

    print(f"\nGenerated Response: {response}")
    assert isinstance(response, str)
    assert response == "This is a dummy response from the MLX model."


    # Restore the original function
    sys.modules[__name__].generate = original_generate
    print("\nTest complete.")
