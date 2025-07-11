import sounddevice as sd
import numpy as np
import soundfile as sf

def record_audio_clip(duration=5, sample_rate=16000):
    """
    Records a short audio clip from the default microphone.

    Args:
        duration (int): The duration of the recording in seconds.
        sample_rate (int): The sample rate for the recording.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The recorded audio as a NumPy array.
            - int: The sample rate of the recording.
    """
    print(f"Recording audio for {duration} seconds...")
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")
        # The recording is a NumPy array. sd.rec returns a column vector, so we flatten it.
        return recording.flatten(), sample_rate
    except Exception as e:
        print(f"An error occurred during audio recording: {e}")
        return None, None

def create_audio_content(audio_data, sample_rate):
    """
    Formats audio data into the structured dictionary for the model.

    Args:
        audio_data (np.ndarray): The audio data.
        sample_rate (int): The sample rate of the audio.

    Returns:
        dict: A dictionary representing the audio part of a multimodal input.
    """
    # Note: The actual key ('audio', 'sampling_rate') might depend on the specific
    # model's processor implementation. This is a sensible default.
    return {'type': 'audio', 'audio': audio_data, 'sampling_rate': sample_rate}


if __name__ == '__main__':
    print("Attempting to record a 5-second audio clip...")
    # Check for available devices
    try:
        print("Available audio devices:", sd.query_devices())
    except Exception as e:
        print(f"Could not query audio devices: {e}")

    audio_clip, sr = record_audio_clip()

    if audio_clip is not None:
        print("Audio recorded successfully.")
        print(f"Shape: {audio_clip.shape}, Sample Rate: {sr}")
        try:
            # Save the captured audio to a file for verification
            output_path = "audio_capture_test.wav"
            sf.write(output_path, audio_clip, sr)
            print(f"Recorded audio saved to {output_path}")

            # Test the formatting function
            formatted_content = create_audio_content(audio_clip, sr)
            print("Formatted content structure created successfully.")
            print("Type:", formatted_content['type'])

        except Exception as e:
            print(f"Error saving audio file: {e}")
    else:
        print("Failed to record audio.")
