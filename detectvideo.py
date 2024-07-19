import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2

# Function to preprocess frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (200, 200))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = frame_rgb / 255.0 
    frame_input = np.expand_dims(frame_array, axis=0)
    return frame_input

# Function to load the trained model
def load_model(model_path):
    try:
        model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')  # Assuming 26 classes
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(model_path)
        model.summary()
        print('Model loaded successfully.')
        return model
    
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

# Function to perform prediction on video frames
def predict_sign_language(model, video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_source}'")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame_input = preprocess_frame(frame)

        # Perform prediction
        prediction = model.predict(frame_input)
        predicted_class = np.argmax(prediction)
        sign_label = sign_classes[predicted_class]

        # Display prediction as text on the frame
        cv2.putText(frame, sign_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with prediction
        cv2.imshow('Sign Language Detection', frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Define the classes
sign_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Main function to load model and start prediction
def main():
    # Path to the trained model
    model_path = 'sign_language_detection_model.keras'  # Replace with your model path
    model = load_model(model_path)
    if model:
        # Replace with the absolute path to your video file
        video_source = 'C:/Users/milan/Documents/PythonProjects/Nullclass/sign/test/video.mp4'
        predict_sign_language(model, video_source)
    else:
        print("Model loading failed. Please check the model path.")

if __name__ == '__main__':
    main()
