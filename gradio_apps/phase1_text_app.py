import gradio as gr
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modules import model_loader, llm_interaction, text_input
from conversation_manager import ConversationManager

# --- Global Variables ---
# These will be loaded once when the app starts
model = None
tokenizer = None
conversation_manager = None
system_prompt_content = ""

# --- Load Model and System Prompt ---
def load_resources():
    global model, tokenizer, conversation_manager, system_prompt_content
    if model is None:
        print("Loading model and tokenizer for Gradio app...")
        model, tokenizer = model_loader.load_model()
        
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "general_reflection.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt_content = f.read()
        
        # Initialize conversation_manager ONLY ONCE here
        conversation_manager = ConversationManager(system_prompt=system_prompt_content)
        print("Resources loaded.")
    return model, tokenizer, conversation_manager

# --- Gradio Chat Function ---
def chat_with_gemma(message, history):
    global model, tokenizer, conversation_manager

    # Ensure resources are loaded
    if model is None:
        model, tokenizer, conversation_manager = load_resources()

    # Clear manager history and rebuild from Gradio's history
    # This is crucial: we clear and rebuild to ensure the manager's internal state
    # matches Gradio's visible history, and to correctly apply system prompt logic
    # only on the *first* actual turn of the conversation.
    conversation_manager.clear_history() # This resets is_first_turn to True

    # Rebuild conversation_manager's history from Gradio's history
    # Gradio's history is [[user1, bot1], [user2, bot2], ...]
    # The current user message ('message' parameter) is NOT yet in 'history'.
    for human_msg, assistant_msg in history:
        if human_msg: # Ensure human message exists
            conversation_manager.add_user_turn([text_input.create_text_content(human_msg)])
        if assistant_msg: # Ensure assistant message exists
            conversation_manager.add_assistant_turn(assistant_msg)

    # Add the current user message
    conversation_manager.add_user_turn([text_input.create_text_content(message)])

    # Get response from LLM
    response = llm_interaction.get_gemma_response(
        model=model,
        tokenizer=tokenizer,
        messages=conversation_manager.get_history(), # Pass the full, correct history
        max_new_tokens=512
    )
    
    # Add assistant turn to history (this will be handled by Gradio's ChatInterface internally
    # when it updates the history, but we also add it to our manager for consistency
    # and for the next turn's history reconstruction).
    conversation_manager.add_assistant_turn(response)

    return response

# --- Gradio Interface ---
if __name__ == "__main__":
    # Load resources once when the app starts
    load_resources()

    gr.ChatInterface(
        fn=chat_with_gemma,
        title="Mind Vault Companion (Text-Only)",
        description="Chat with the Gemma 3n AI companion. It focuses on helping you reflect on your thoughts and feelings.",
        examples=[
            "I'm feeling a bit down today.",
            "I had a really productive day at work.",
            "I'm not sure how to feel about this situation."
        ],
        retry_btn=None,
        undo_btn="Delete Last Turn",
        clear_btn="Clear All",
    ).launch()