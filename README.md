# Mind Vault Companion: Multimodal Desktop Prototype (MLX Version)

## Project Overview

The **Mind Vault Companion** is an innovative mental well-being application designed to offer private, accessible, and non-dependent support. Leveraging the capabilities of Google's **Gemma 3n** on-device AI model, this companion provides a safe space for users to check in with their emotions across multiple modalities (text, video, audio). It focuses on understanding subtle emotional nuances and fostering self-awareness without promoting dependency on the AI.

This project uses the **MLX framework** for efficient local inference on Apple silicon, ensuring a responsive and private user experience.

## Core Concept

The primary goal is to create a digital mental health companion that:
* **Facilitates Multimodal Check-ins:** Allows users to log their feelings and thoughts through text, webcam video (facial expressions, demeanor), and microphone audio (voice tone, speech content).
* **Contextual Understanding:** Utilizes Gemma 3n's ability to maintain conversation history to provide relevant and empathetic responses.
* **Emotional Nuance Detection:** Leverages Gemma 3n's native multimodal encoders to implicitly understand emotional states from various inputs.
* **Promotes Non-Dependency:** Designed to empower users to develop their own coping mechanisms and self-awareness, rather than becoming reliant on the AI for direct advice. It focuses on reflective questions and observations.
* **Offline-Capable Design:** The use of MLX and a local model ensures the application can run entirely offline, maximizing privacy and accessibility.

## Technology Stack

* **Large Language Model:** Gemma 3n (e.g., `mlx-community/gemma-3n-E2B-it-4bit`)
* **LLM Framework:** Apple MLX (`mlx`, `mlx-lm`, `mlx-vlm`)
* **Programming Language:** Python 3
* **User Interface:** Gradio
* **Video Processing:** OpenCV (`cv2`)
* **Audio Processing:** `sounddevice`, `soundfile`, `librosa`
* **Data Handling:** `pandas`, Hugging Face `datasets`
* **Evaluation Metrics:** `scikit-learn`, `sentence-transformers`

---

## **Project Structure**
MindVault/
├── .venv/                          # Python Virtual Environment
├── src/                            # All application source code
│   ├── modules/                    # Core functional modules
│   │   ├── model_loader.py         # Loads Gemma 3n model and tokenizer using MLX
│   │   ├── llm_interaction.py      # Encapsulates MLX-based response generation
│   │   ├── text_input.py           # Handles raw text input processing
│   │   ├── video_input.py          # Handles webcam frame capture
│   │   └── audio_input.py          # Handles microphone audio capture
│   │
│   ├── prompts/                    # Directory for all chat prompt templates
│   │   ├── general_reflection.txt
│   │   ├── video_analysis_prompt.txt
│   │   └── audio_analysis_prompt.txt
│   │
│   ├── conversation_manager.py     # Manages the conversation history
│   └── main_console_app.py         # Console-based app for testing
│
├── models/                         # Local directory for storing MLX models
│   └── mlx-community/
│       └── gemma-3n-E2B-it-4bit/
│
├── scripts/                        # Utility scripts
│   └── download_model.py           # Downloads the MLX model from Hugging Face
│
├── gradio_apps/                    # Gradio interface files
│   └── phase1_text_app.py          # Gradio app for Text-only interaction
│
├── .gitignore                      # Standard git ignore file
├── requirements.txt                # List of all Python dependencies
└── README.md                       # This file

## **Setup and Execution**

### **1. Environment Setup**

1.  **Clone the repository and navigate to the project directory.**
2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Log in to Hugging Face Hub** (required to download the model):
    ```bash
    huggingface-cli login
    ```

### **2. Download the Model**

Run the download script to fetch the MLX-quantized Gemma 3n model from the Hugging Face Hub and store it in the local `models/` directory.

```bash
python scripts/download_model.py
```

This will create the following structure: `models/mlx-community/gemma-3n-E2B-it-4bit`.

### **3. Run the Application**

You can run either the console-based application for quick tests or the Gradio web interface for a richer user experience.

*   **Console Application:**
    ```bash
    python src/main_console_app.py
    ```

*   **Gradio Web App:**
    ```bash
    python gradio_apps/phase1_text_app.py
    ```
    This will launch a local web server. Open the provided URL in your browser to interact with the Mind Vault Companion.

---
