import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Function to preprocess frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (200, 200))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = img_to_array(frame_rgb) / 255.0
    frame_input = np.expand_dims(frame_array, axis=0)
    return frame_input

# Function to load the trained model
def load_model(model_path):
    print('loading model...')#
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading the model: {e}")

    print('model loaded')#
    return model

# Function to perform real-time prediction on video frames
def predict_sign_language(model, video_source):
    print('starting prediction...')#
    cap = cv2.VideoCapture(video_source)
    print('video capture enabled')#
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_input = preprocess_frame(frame)

        prediction = model.predict(frame_input)
        predicted_class = np.argmax(prediction)

        sign_label = sign_classes[predicted_class]
        cv2.putText(frame, sign_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(sign_label)

        cv2.imshow('Sign Language Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Define the classes
sign_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Main function to load model and start prediction
def main():
    print('starting...')
    model_path = 'sign_language_detection_model.keras'  # Replace with your model path
    model = load_model(model_path)
    predict_sign_language(model, video_source=0)

if __name__ == '__main__':
    main()
