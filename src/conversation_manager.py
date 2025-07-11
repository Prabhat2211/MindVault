"""
Manages the conversation history for the Mind Vault Companion.
"""

class ConversationManager:
    """
    Manages conversation history, including adding turns and truncation.
    """
    def __init__(self, system_prompt=None, max_history_turns=10):
        """
        Initializes the ConversationManager.

        Args:
            system_prompt (str, optional): The initial system prompt.
            max_history_turns (int): The maximum number of conversational turns
                                     (user + assistant) to keep in history.
        """
        self.system_prompt_content = [{'type': 'text', 'text': system_prompt}] if system_prompt else None
        self.max_history_messages = max_history_turns * 2
        self.history = []
        self.is_first_turn = True

    def add_user_turn(self, content):
        """
        Adds a user's turn to the conversation history.

        On the first turn, it prepends the system prompt to the user's content.

        Args:
            content (list): A list of dictionaries representing the multimodal input.
        """
        final_content = content
        if self.is_first_turn and self.system_prompt_content:
            # Prepend system prompt to the first user message
            final_content = self.system_prompt_content + content
            self.is_first_turn = False  # Unset the flag

        self.history.append({'role': 'user', 'content': final_content})
        self.truncate_history()

    def add_assistant_turn(self, text):
        """
        Adds an assistant's turn to the conversation history.
        """
        content = [{'type': 'text', 'text': text}]
        self.history.append({'role': 'assistant', 'content': content})
        self.truncate_history()

    def get_history(self):
        """
        Returns the current conversation history.
        """
        return self.history

    def truncate_history(self):
        """
        Truncates the history to ensure it doesn't exceed the maximum length.
        """
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages:]

    def clear_history(self):
        """
        Clears the entire conversation history and resets the first turn flag.
        """
        self.history = []
        self.is_first_turn = True

if __name__ == '__main__':
    # Example Usage
    manager = ConversationManager(max_history_turns=2)
    
    print("Initial History:", manager.get_history())

    # Turn 1
    manager.add_user_turn([{'type': 'text', 'text': 'Hello, how are you?'}])
    manager.add_assistant_turn("I am an AI assistant. I am doing well.")
    print("\nHistory after Turn 1:", manager.get_history())
    print("History length:", len(manager.get_history()))


    # Turn 2
    manager.add_user_turn([{'type': 'text', 'text': 'What can you do?'}])
    manager.add_assistant_turn("I can help you with a variety of tasks.")
    print("\nHistory after Turn 2:", manager.get_history())
    print("History length:", len(manager.get_history()))

    # Turn 3 (this should cause truncation)
    manager.add_user_turn([{'type': 'text', 'text': 'Tell me a joke.'}])
    manager.add_assistant_turn("Why don't scientists trust atoms? Because they make up everything!")
    print("\nHistory after Turn 3 (truncation should occur):")
    print(manager.get_history())
    print("History length:", len(manager.get_history()))

    # Clear history
    manager.clear_history()
    print("\nHistory after clearing:", manager.get_history())
