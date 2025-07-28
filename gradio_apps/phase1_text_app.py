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
system_prompt_content = ""

# --- Load Model and System Prompt ---
def load_resources():
    global model, tokenizer, system_prompt_content
    if model is None:
        print("Loading model and tokenizer for Gradio app...")
        model, tokenizer = model_loader.load_model()
        
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "general_reflection.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt_content = f.read()
        
        print("Resources loaded.")

# --- Gradio Chat Function ---
def chat_with_gemma(message, history, conversation_state):
    global model, tokenizer, system_prompt_content

    # If it's the start of a new chat, initialize a new ConversationManager
    if conversation_state is None:
        conversation_state = ConversationManager(system_prompt=system_prompt_content)

    # Add the current user message
    conversation_state.add_user_turn([text_input.create_text_content(message)])

    # Get response from LLM
    response = llm_interaction.get_gemma_response(
        model=model,
        tokenizer=tokenizer,
        messages=conversation_state.get_history(), # Pass the session-specific history
        max_new_tokens=512
    )
    
    # Add assistant turn to our session-specific history
    conversation_state.add_assistant_turn(response)

    # The first return value is the response to be displayed
    # The second return value is the updated state to be preserved
    return response, conversation_state

# --- Gradio Interface ---
if __name__ == "__main__":
    # Load resources once when the app starts
    load_resources()

    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
        gr.Markdown("""
        # Mind Vault Companion (Text-Only)
        Chat with the Gemma 3n AI companion. It focuses on helping you reflect on your thoughts and feelings.
        """)
        
        # Session state to store the ConversationManager instance per user
        conversation_state = gr.State()

        chatbot = gr.Chatbot(height=500)
        
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter your message and press Enter",
                container=False,
                scale=7,
            )
            submit_btn = gr.Button("Submit", variant="primary", scale=1)

        gr.Examples(
            examples=[
                "I'm feeling a bit down today.",
                "I had a really productive day at work.",
                "I'm not sure how to feel about this situation."
            ],
            inputs=msg
        )

        def respond(message, chat_history, state):
            bot_message, updated_state = chat_with_gemma(message, chat_history, state)
            chat_history.append((message, bot_message))
            return "", chat_history, updated_state

        msg.submit(respond, [msg, chatbot, conversation_state], [msg, chatbot, conversation_state])
        submit_btn.click(respond, [msg, chatbot, conversation_state], [msg, chatbot, conversation_state])

    demo.launch()