
"""
Main console application for the Mind Vault Companion (MLX version).
"""
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.modules import model_loader
from src.modules import llm_interaction
from src.modules import text_input
from src.conversation_manager import ConversationManager

def main():
    """
    Main function to run the console-based chat application with MLX.
    """
    print("--- Mind Vault Companion (Console Mode | MLX) ---")
    print("Initializing... This may take a while as the model is loaded.")

    try:
        # Load the MLX model and tokenizer
        model, tokenizer = model_loader.load_model()

        # Load the system prompt
        prompt_path = Path(__file__).parent / "prompts" / "general_reflection.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        manager = ConversationManager(system_prompt=system_prompt)

    except Exception as e:
        print(f"Failed to initialize the application: {e}")
        return

    print("Initialization complete. You can start chatting.")
    print("Type 'quit' or 'exit' to end the conversation.")

    while True:
        try:
            user_text = input("\nYou: ")

            if user_text.lower() in ['quit', 'exit']:
                print("Ending conversation. Goodbye!")
                break

            # 1. Add user turn
            user_content_list = [text_input.create_text_content(user_text)]
            manager.add_user_turn(user_content_list)

            # 2. Get response from LLM
            print("\nCompanion: Thinking...")
            assistant_response = llm_interaction.get_gemma_response(
                model=model,
                tokenizer=tokenizer,
                messages=manager.get_history(),
                max_new_tokens=512
            )

            # 3. Add assistant turn to history
            manager.add_assistant_turn(assistant_response)

            # 4. Print response
            print(f"\nCompanion: {assistant_response}")

        except KeyboardInterrupt:
            print("\nEnding conversation. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
