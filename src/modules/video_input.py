import cv2
from PIL import Image

def capture_webcam_frame():
    """
    Captures a single frame from the default webcam.

    Returns:
        PIL.Image.Image or None: A PIL Image of the captured frame in RGB format,
                                 or None if the capture failed.
    """
    # 0 is the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Read a single frame
    ret, frame = cap.read()

    # Release the camera resource
    cap.release()

    if not ret:
        print("Error: Could not read frame from webcam.")
        return None

    # OpenCV captures in BGR format, convert it to RGB for PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a PIL Image
    pil_image = Image.fromarray(rgb_frame)

    return pil_image

def create_image_content(image):
    """
    Formats a PIL Image into the structured dictionary for the model.

    Args:
        image (PIL.Image.Image): The image to format.

    Returns:
        dict: A dictionary representing the image part of a multimodal input.
    """
    return {'type': 'image', 'image': image}

if __name__ == '__main__':
    print("Attempting to capture a frame from the webcam...")
    captured_image = capture_webcam_frame()

    if captured_image:
        print("Frame captured successfully.")
        try:
            # Save the captured image to a file for verification
            output_path = "webcam_capture_test.jpg"
            captured_image.save(output_path)
            print(f"Captured image saved to {output_path}")
            
            # Test the formatting function
            formatted_content = create_image_content(captured_image)
            print("Formatted content structure created successfully.")
            print("Type:", formatted_content['type'])

        except Exception as e:
            print(f"Error saving image: {e}")
    else:
        print("Failed to capture frame.")
