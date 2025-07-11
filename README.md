# Mind Vault Companion: Multimodal Desktop Prototype

## Project Overview

The **Mind Vault Companion** is an innovative mental well-being application designed to offer private, accessible, and non-dependent support. Leveraging the capabilities of Google's **Gemma 3n** on-device AI model (accessed via Hugging Face `transformers` for local inference), this companion aims to provide a safe space for users to check in with their emotions across multiple modalities (text, video, audio). It focuses on understanding subtle emotional nuances and fostering self-awareness without promoting dependency on the AI.

This project outlines a phased development approach to build a robust, modular, and evaluable multimodal prototype, culminating in an interactive Gradio web interface.

## Core Concept

The primary goal is to create a digital mental health companion that:
* **Facilitates Multimodal Check-ins:** Allows users to log their feelings and thoughts through text, webcam video (facial expressions, demeanor), and microphone audio (voice tone, speech content).
* **Contextual Understanding:** Utilizes Gemma 3n's ability to maintain conversation history to provide relevant and empathetic responses.
* **Emotional Nuance Detection:** Leverages Gemma 3n's native multimodal encoders to implicitly understand emotional states from various inputs.
* **Promotes Non-Dependency:** Designed to empower users to develop their own coping mechanisms and self-awareness, rather than becoming reliant on the AI for direct advice. It focuses on reflective questions and observations.
* **Offline-Capable Design:** While the prototype runs on a desktop, the underlying Gemma 3n model and architecture are chosen for their suitability for future on-device, offline mobile deployment.

## Technology Stack

* **Large Language Model:** Gemma 3n (specifically the `-it` instruction-tuned variants, e.g., `E2B-it` or `E4B-it`)
* **LLM Framework:** Hugging Face `transformers`
* **Local Inference Backend:** PyTorch (running on CUDA/GPU for optimal performance, or CPU)
* **Programming Language:** Python 3
* **User Interface:** Gradio (for interactive web-based prototypes)
* **Video Processing:** OpenCV (`cv2`)
* **Audio Processing:** `sounddevice`, `soundfile`, `librosa`
* **Data Handling:** `pandas`, Hugging Face `datasets`
* **Evaluation Metrics:** `scikit-learn`, `sentence-transformers`
* **Fine-tuning (Future):** `peft`, `trl`, `accelerate`, `bitsandbytes`

---

## **Project Structure**
Hf/
├── .venv/                          # Python Virtual Environment
├── src/                            # All application source code
│   ├── modules/                    # Core functional modules
│   │   ├── model_loader.py         # Loads Gemma 3n model, processor, tokenizer
│   │   ├── llm_interaction.py      # Encapsulates get_gemma_multimodal_response (with timing)
│   │   ├── text_input.py           # Handles raw text input processing
│   │   ├── video_input.py          # Handles webcam frame capture, PIL Image conversion
│   │   └── audio_input.py          # Handles microphone audio capture, NumPy array conversion
│   │
│   ├── prompts/                    # Directory for all chat prompt templates (.txt files)
│   │   ├── general_reflection.txt  # Main companion prompt
│   │   ├── video_analysis_prompt.txt # Prompt for specific video-based insights
│   │   └── audio_analysis_prompt.txt # Prompt for specific audio-based insights
│   │
│   ├── conversation_manager.py     # Manages the conversation history (new!)
│   ├── main_console_app.py         # Console-based app for basic testing (optional, but good for debugging core logic)
│   └── live_video_app.py           # (Phase 2) Handles continuous live video detection and display
│
├── data/                           # Datasets for evaluation and fine-tuning
│   ├── raw/                        # Raw downloaded datasets
│   │   ├── FER2013/                # Facial Emotion Recognition Images
│   │   ├── RAVDESS/                # Audio Emotion Recognition (Speech)
│   │   └── Text_Emotion_Dataset/   # e.g., dair-ai/emotion text data
│   │
│   ├── processed/                  # Processed datasets (CSV/JSONL, ready for use)
│   │   ├── eval_images.csv         # Map image paths to labels for evaluation
│   │   ├── eval_audio.csv          # Map audio paths to labels for evaluation
│   │   └── eval_text.csv           # Map text snippets to labels for evaluation
│   │
│   └── finetuning_data/            # Data specifically formatted for fine-tuning
│       └── gemma_finetune_ds.jsonl # Combined multimodal data for fine-tuning
│
├── scripts/                        # Utility scripts
│   ├── data_prep.py                # Prepares raw datasets into 'processed/' format
│   └── finetune_gemma3n.py         # Script for fine-tuning the model
│
├── eval_prompts/                   # Specific prompts only used for automated evaluation
│   ├── eval_image_emotion_template.txt
│   ├── eval_audio_emotion_template.txt
│   └── eval_text_emotion_template.txt
│
├── gradio_apps/                    # Gradio interface files for each phase
│   ├── phase1_text_app.py          # Gradio app for Phase 1 (Text only)
│   ├── phase2_video_app.py         # Gradio app for Phase 2 (Video input added)
│   └── phase3_audio_app.py         # Gradio app for Phase 3 (Audio input added, combined multimodal)
│
├── evaluation_results/             # Stores evaluation output, logs
│   ├── phase1_text_eval_summary.txt
│   ├── phase1_text_results.csv
│   ├── phase2_video_eval_summary.txt
│   ├── phase2_video_results.csv
│   ├── phase3_audio_eval_summary.txt
│   └── phase3_audio_results.csv
│
├── models/                         # (Optional) For saving your fine-tuned Gemma 3n model
│   └── finetuned_gemma3n/
│
├── .gitignore                      # Standard git ignore file
├── requirements.txt                # List of all Python dependencies
└── README.md                       # This file

## **Development Phases**

### **Phase 0: Foundation & Core Modules**

**Objective:** Set up the project environment, install all dependencies, and establish core, reusable modules for model loading, interaction, and input handling.

* **Setup Steps:**
    1.  Open your terminal/command prompt within `Hf` folder.
    2.  Create and activate a Python virtual environment:
        * `python -m venv .venv`
        * **macOS/Linux:** `source ./.venv/bin/activate`
    3.  Log in to Hugging Face Hub: `huggingface-cli login` (essential to access Gemma 3n models after accepting their terms).
* **Core Modules Implementation (`src/modules/`):**
    * **`model_loader.py`:** Implement the function to load `AutoProcessor`, `AutoTokenizer`, and `Gemma3nForConditionalGeneration` (e.g., `google/gemma-3n-E2B-it`) onto the appropriate device (CUDA/CPU).
    * **`llm_interaction.py`:** Implement `get_gemma_multimodal_response` function, which includes:
        * Taking `conversation_history`, `new_user_content` (list of dicts for multimodal parts), and `max_new_tokens`.
        * Utilizing `time.time()` to measure and print **inference time** for the entire process (from input preparation to response decoding).
        * Returning Gemma 3n's text response.
    * **`text_input.py`:** Implement basic functions for processing raw text inputs.
    * **`video_input.py`:** Implement `capture_webcam_frame()` to capture a single PIL Image frame from the webcam.
    * **`audio_input.py`:** Implement `record_audio_clip()` for short audio recordings from the microphone (returning NumPy array and sample rate).
    * **`conversation_manager.py`:** Implement functions to `add_user_turn`, `add_assistant_turn`, `get_history`, and `truncate_history` (to manage context window limits and maintain conversational flow).
* **`prompts/`:** Create this directory and add initial `.txt` files for general prompt templates (e.g., `general_reflection.txt`).
* **`main_console_app.py`:** Create a basic console application using the new modules to confirm model loading and simple text-based interaction. This is for quick internal debugging.

---

### **Phase 1: Text-Only Companion**

**Objective:** Build a functional text-based companion with conversation history, establish the text-emotion recognition evaluation pipeline, and create a Gradio UI for text interaction.

* **1.1 Development - Console & Gradio App:**
    * Refine `src/main_console_app.py` to fully implement the text-based conversation loop, utilizing `conversation_manager` for history management and `llm_interaction` for responses.
    * Create `gradio_apps/phase1_text_app.py`:
        * Load Gemma 3n components via `model_loader`.
        * Set up a `gr.ChatInterface` or `gr.Interface` with `gr.Textbox` for user input and a text output component.
        * Maintain conversation history within Gradio's state management (often passed via the `gr.State` component).
        * Implement the Gradio function to process text input, update history, call `llm_interaction.get_gemma_multimodal_response` (with text-only content), and display Gemma's response.
* **1.2 Evaluation - Text Emotion Recognition:**
    * **Data Acquisition:** Download a text emotion dataset (e.g., `dair-ai/emotion` from Hugging Face Datasets) into `data/raw/Text_Emotion_Dataset/`.
    * **`scripts/data_prep.py`:** Write a script to process the raw text dataset into `data/processed/eval_text.csv` (e.g., columns: `text_snippet`, `emotion_label`).
    * **`eval_prompts/eval_text_emotion_template.txt`:** Create a concise prompt instructing Gemma 3n to identify the primary emotion from a given text. Example: "Analyze the emotion of this text: '[TEXT]'. Respond with one word: [happy, sad, angry, neutral, surprise, fear, disgust]."
    * **`scripts/evaluate_model.py`:**
        * Extend this script to:
            * Load `data/processed/eval_text.csv`.
            * Loop through each text snippet, format messages with `eval_text_emotion_template.txt` and `conversation_manager` (for the prompt context).
            * Call `llm_interaction.get_gemma_multimodal_response` for inference.
            * **Measure and record inference time** for each evaluation sample.
            * Normalize Gemma's textual output to discrete emotion labels.
            * Calculate classification metrics (accuracy, precision, recall, F1-score) using `scikit-learn`.
            * Store detailed results (including inference times) in `evaluation_results/phase1_text_results.csv` and a summary in `evaluation_results/phase1_text_eval_summary.txt`.
* **1.3 Testing & Refinement:**
    * Thoroughly test `main_console_app.py` for conversational flow and context.
    * Launch and test `gradio_apps/phase1_text_app.py` for user experience and responsiveness (observing inference times).
    * Analyze initial evaluation results to identify areas for prompt refinement.

---

### **Phase 2: Adding Video Input**

**Objective:** Integrate webcam video input for emotional insights, establish the video-emotion recognition evaluation pipeline, and update the Gradio UI to support video interaction.

* **2.1 Development - Gradio App & Live Video:**
    * **`src/modules/video_input.py`:** Ensure `capture_webcam_frame()` is robust.
    * **`gradio_apps/phase2_video_app.py`:**
        * Copy `phase1_text_app.py` and modify.
        * Add `gr.Image(sources=["webcam"])` as an input component.
        * Update the Gradio function logic to:
            * Handle multimodal input (image + text) using `llm_interaction.py`.
            * Prepare `new_user_content` with the PIL Image and text input, and pass it to `llm_interaction.get_gemma_multimodal_response`.
            * Refine `prompts/video_analysis_prompt.txt` to guide Gemma 3n on interpreting visual cues and relating them to emotional states within the conversation context.
    * **`src/live_video_app.py`:** Develop the continuous live video feed application. This will use OpenCV for capture, multithreading for background Gemma 3n inference (processing frames periodically via `llm_interaction.py`), and overlay textual insights (with their inference times) on the live video stream.
* **2.2 Evaluation - Video Emotion Recognition:**
    * **Data Acquisition:** Download a facial emotion dataset (e.g., FER2013) into `data/raw/FER2013/`.
    * **`scripts/data_prep.py`:** Extend to process the raw image dataset into `data/processed/eval_images.csv` (mapping image file paths to emotion labels).
    * **`eval_prompts/eval_image_emotion_template.txt`:** Create a prompt tailored for image-based emotion classification. Example: "Analyze the facial expression in this image. Which single emotion (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust) is most prominent? Respond only with the emotion word."
    * **`scripts/evaluate_model.py`:**
        * Extend to:
            * Load `eval_images.csv`.
            * Loop through images, prepare multimodal messages (image + `eval_image_emotion_template.txt` using `conversation_manager`).
            * Call `llm_interaction.get_gemma_multimodal_response` for inference.
            * **Measure and record inference time** for each evaluation sample.
            * Map Gemma's textual output to discrete emotion labels.
            * Calculate and store classification metrics in `evaluation_results/phase2_video_results.csv` and a summary in `evaluation_results/phase2_video_eval_summary.txt`.
* **2.3 Testing & Refinement:**
    * Test video input and combined text/video input in `gradio_apps/phase2_video_app.py` (observing inference times).
    * Thoroughly test `src/live_video_app.py` for performance and accuracy of live insights.
    * Refine video-specific prompts based on evaluation results.

---

### **Phase 3: Adding Audio Input**

**Objective:** Integrate microphone audio input for emotional insights, establish the audio-emotion recognition evaluation pipeline, and create a final, combined Gradio UI.

* **3.1 Development - Gradio App:**
    * **`src/modules/audio_input.py`:** Ensure `record_audio_clip()` is robust.
    * **`gradio_apps/phase3_audio_app.py`:**
        * Copy `phase2_video_app.py` and modify.
        * Add `gr.Audio(sources=["microphone"])` as an input component.
        * Update the Gradio function logic to:
            * Handle full multimodal input (audio + text + optional image).
            * Pass the captured audio (NumPy array with sampling rate) to `llm_interaction.get_gemma_multimodal_response` within the `new_user_content` list.
            * Refine `prompts/audio_analysis_prompt.txt` for guiding Gemma 3n on interpreting voice tone, speech content, and their emotional implications within the conversation context.
* **3.2 Evaluation - Audio Emotion Recognition:**
    * **Data Acquisition:** Download an audio emotion dataset (e.g., RAVDESS) into `data/raw/RAVDESS/`.
    * **`scripts/data_prep.py`:** Extend to process the raw audio dataset into `data/processed/eval_audio.csv` (mapping audio file paths to emotion labels and sampling rates).
    * **`eval_prompts/eval_audio_emotion_template.txt`:** Create a prompt for audio-based emotion classification. Example: "Analyze the emotion conveyed in this speech clip. Which single emotion (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust) is most prominent? Respond only with the emotion word."
    * **`scripts/evaluate_model.py`:**
        * Extend to:
            * Load `eval_audio.csv`.
            * Loop through audio files, prepare multimodal messages (audio + `eval_audio_emotion_template.txt` using `conversation_manager`).
            * Call `llm_interaction.get_gemma_multimodal_response` for inference.
            * **Measure and record inference time** for each evaluation sample.
            * Map Gemma's textual output to discrete emotion labels.
            * Calculate and store classification metrics in `evaluation_results/phase3_audio_results.csv` and a summary in `evaluation_results/phase3_audio_eval_summary.txt`.
* **3.3 Testing & Refinement:**
    * Thoroughly test audio input and combined multimodal inputs in `gradio_apps/phase3_audio_app.py` (observing inference times).
    * Refine audio-specific prompts based on evaluation results.

---

### **Beyond Phases: Fine-tuning & Finalization**

* **Fine-tuning (Optional, but powerful):**
    * **Objective:** Enhance Gemma 3n's performance for specific emotional nuances and desired companion-like response styles.
    * **`data/finetuning_data/`:** Curate or create a custom multimodal dataset (text, image, audio, combined) with desired Gemma 3n responses. This is a significant data annotation effort.
    * **`scripts/finetune_gemma3n.py`:** Implement the fine-tuning script using `peft`, `trl`, `accelerate`, and `bitsandbytes` for efficient LoRA/QLoRA.
    * **Evaluation:** Re-run evaluations from Phase 1-3 with the fine-tuned model to quantify improvements.
* **Final Gradio App:**
    * Consolidate `gradio_apps/phase3_audio_app.py` into a single, comprehensive `gradio_apps/app.py`.
    * Ensure all multimodal inputs work seamlessly and conversation history is robust.
    * Add any final UI polish.
* **Documentation & Competition Submission:**
    * **`README.md`:** Keep this file updated with all current features, setup, usage instructions, evaluation summary, and future vision.
    * **Video Demo:** Plan and record a compelling video demonstrating all key features, especially the live multimodal interaction and the companion's contextual understanding.
    * **Technical Write-up:** Detail your modular architecture, Gemma 3n integration, multimodal processing, fine-tuning strategy (if applicable), and comprehensive evaluation methodology and results.

---