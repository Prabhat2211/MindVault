import gradio as gr
import sys
from pathlib import Path
from PIL import Image

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modules import model_loader, llm_interaction, text_input
from conversation_manager import ConversationManager

# --- Global Variables ---
model = None
tokenizer = None
system_prompt_content = ""
video_prompt_content = ""
emotion_prompt_content = ""

# --- Load Model and System Prompt ---
def load_resources():
    global model, tokenizer, system_prompt_content, video_prompt_content, emotion_prompt_content
    if model is None:
        print("Loading model and tokenizer for Gradio app...")
        model, tokenizer = model_loader.load_model()
        
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "general_reflection.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt_content = f.read()

        video_prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "video_analysis_prompt.txt"
        with open(video_prompt_path, 'r', encoding='utf-8') as f:
            video_prompt_content = f.read()

        emotion_prompt_path = Path(__file__).parent.parent / "eval_prompts" / "eval_image_emotion_template.txt"
        with open(emotion_prompt_path, 'r', encoding='utf-8') as f:
            emotion_prompt_content = f.read()
        
        print("Resources loaded.")

# --- Gradio Chat Function ---
def chat_with_gemma(message, image, history, conversation_state):
    global model, tokenizer, system_prompt_content, video_prompt_content, emotion_prompt_content

    if conversation_state is None:
        conversation_state = ConversationManager(system_prompt=system_prompt_content)

    # --- Main Conversational Response ---
    full_message = f"{video_prompt_content}\n\n{message}" if image else message
    conversation_state.add_user_turn([text_input.create_text_content(full_message)])

    response = llm_interaction.get_gemma_response(
        model=model,
        tokenizer=tokenizer,
        messages=conversation_state.get_history(),
        image=image,
        max_new_tokens=512
    )
    
    conversation_state.add_assistant_turn(response)

    # --- Emotion Analysis ---
    emotion = ""
    if image:
        emotion_messages = [{"role": "user", "content": [text_input.create_text_content(emotion_prompt_content)]}]
        emotion = llm_interaction.get_gemma_response(
            model=model,
            tokenizer=tokenizer,
            messages=emotion_messages,
            image=image,
            max_new_tokens=32
        )

    return response, conversation_state, emotion

# --- Gradio Interface ---
if __name__ == "__main__":
    load_resources()

    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
        gr.Markdown("""
        # Mind Vault Companion (Video-Enabled)
        Share a snapshot from your webcam along with your thoughts.
        """)
        
        conversation_state = gr.State()
        chatbot = gr.Chatbot(height=400)
        emotion_label = gr.Label(label="Detected Emotion")
        
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(sources=["webcam"], type="pil")
            with gr.Column(scale=2):
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

        def respond(message, image, chat_history, state):
            bot_message, updated_state, emotion = chat_with_gemma(message, image, chat_history, state)
            chat_history.append((message, bot_message))
            return "", None, chat_history, updated_state, emotion

        inputs = [msg, img, chatbot, conversation_state]
        outputs = [msg, img, chatbot, conversation_state, emotion_label]

        submit_btn.click(respond, inputs, outputs)
        msg.submit(respond, inputs, outputs)

    demo.launch()