import cv2
import sys
import time
from pathlib import Path
from PIL import Image

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.modules import model_loader, llm_interaction, text_input, video_input
from src.conversation_manager import ConversationManager

# --- Global Variables ---
model = None
tokenizer = None
conversation_manager = None
latest_insight = "Initializing..."
last_analysis_time = 0
ANALYSIS_INTERVAL = 10 # seconds

def analyze_frame(frame):
    """
    Function to run Gemma inference sequentially.
    """
    global latest_insight
    
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Prepare content for the model
    video_prompt_path = Path(__file__).parent / "prompts" / "video_analysis_prompt.txt"
    with open(video_prompt_path, 'r', encoding='utf-8') as f:
        video_prompt = f.read()

    user_content = [
        video_input.create_image_content(pil_image),
        text_input.create_text_content(video_prompt)
    ]
    
    conversation_manager.clear_history()
    conversation_manager.add_user_turn(user_content)

    # Get response
    insight = llm_interaction.get_gemma_response(
        model=model,
        tokenizer=tokenizer,
        messages=conversation_manager.get_history(),
        max_new_tokens=128
    )
    
    latest_insight = insight

def main():
    global model, tokenizer, conversation_manager, last_analysis_time

    print("--- Live Video Analysis --- ")
    print("Loading model...")
    model, tokenizer = model_loader.load_model()
    
    # Load general reflection prompt
    prompt_path = Path(__file__).parent / "prompts" / "general_reflection.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    conversation_manager = ConversationManager(system_prompt=system_prompt)
    print("Model loaded. Starting video stream...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_analysis_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_analysis_time >= ANALYSIS_INTERVAL:
            analyze_frame(frame.copy())
            last_analysis_time = time.time()

        # Display the latest insight on the frame
        cv2.putText(frame, f"Insight: {latest_insight}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Live Video Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()