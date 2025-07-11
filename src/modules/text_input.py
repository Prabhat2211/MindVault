"""
Handles processing of raw text inputs for the Mind Vault Companion.
"""

def create_text_content(text):
    """
    Formats a raw text string into the structured dictionary for the model.

    Args:
        text (str): The raw text input from the user.

    Returns:
        dict: A dictionary representing the text part of a multimodal input.
    """
    return {'type': 'text', 'text': text}

if __name__ == '__main__':
    text_input = "Hello, world! This is a test."
    formatted_content = create_text_content(text_input)
    print("Original text:", text_input)
    print("Formatted content:", formatted_content)
    assert formatted_content == {'type': 'text', 'text': text_input}
    print("Test passed.")
