import time
import torch

def get_gemma_multimodal_response(model, processor, tokenizer, messages, max_new_tokens=256):
    """
    Generates a multimodal response from the Gemma 3n model.

    Args:
        model: The loaded Gemma 3n model.
        processor: The loaded model processor.
        tokenizer: The loaded model tokenizer.
        messages (list): The conversation history, formatted as a list of
                         dictionaries, e.g., [{'role': 'user', 'content': [...]}]
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated text response from the model.
    """
    # 1. Start timer
    start_time = time.time()

    # 2. Prepare inputs
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    images = [part['image'] for turn in messages for part in turn['content'] if part['type'] == 'image']

    inputs = processor(
        text=prompt,
        images=images if images else None,
        return_tensors="pt"
    ).to(model.device)

    # 3. Generate response
    generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 4. Decode the response
    input_token_len = inputs['input_ids'].shape[1]
    response_text = tokenizer.decode(generate_ids[0][input_token_len:], skip_special_tokens=True)

    # 5. Stop timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"(Inference time: {duration:.2f} seconds)")

    return response_text

if __name__ == '__main__':
    # This is a placeholder for a test.
    # To run a full test, we would need to load the model, which is slow.
    # We assume the model is loaded and create dummy components for this example.
    
    print("Testing llm_interaction module...")

    # Dummy components for demonstration purposes
    class DummyModel:
        def __init__(self, device='cpu'):
            self.device = device
        def generate(self, **kwargs):
            # Return dummy IDs that represent a response
            input_ids = kwargs['input_ids']
            # Simulate adding new tokens to the end
            response_tokens = torch.tensor([[100, 200, 300]]) # Dummy token IDs for the response
            full_sequence = torch.cat([input_ids, response_tokens], dim=1)
            return full_sequence
    
    class DummyTokenizer:
        def apply_chat_template(self, conv, tokenize, add_generation_prompt):
            return "<start_of_turn>user\nWhat is in this image?<end_of_turn>\n<start_of_turn>model\n"
        def decode(self, ids, skip_special_tokens):
            return "This is a dummy response from the model."

    class DummyProcessor:
        def __call__(self, text, images, return_tensors):
            # Return dummy input_ids for the text prompt
            return {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), 'attention_mask': torch.tensor([[1]*8])}

    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()
    dummy_processor = DummyProcessor()

    # Example conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "image": "dummy_image_placeholder"} # Placeholder
            ]
        }
    ]

    print(f"\nGenerating response for: {messages[0]['content'][0]['text']}")
    
    response = get_gemma_multimodal_response(
        dummy_model,
        dummy_processor,
        dummy_tokenizer,
        messages,
        max_new_tokens=50
    )

    print(f"\nGenerated Response: {response}")
    print("\nTest complete.")