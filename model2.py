import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def train(model, train_generator, val_generator, batch_size):
    try:
        epochs = 5

        history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // batch_size,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=val_generator.samples // batch_size
        )

        # Evaluate the model on the validation set
        val_loss, val_acc = model.evaluate(val_generator, verbose=1)
        print(f"Validation accuracy: {val_acc:.2f}")

        # Save the model
        model.save('sign_language_detection_model.keras')
    
    except Exception as e:
        print(f"Error during training: {e}")

def train1(model, model_path, train_generator, val_generator, batch_size):
    try:
        model = tf.keras.models.load_model(model_path)
        opt=Adam(learning_rate=0.00001)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        epochs = 5

        history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // batch_size,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=val_generator.samples // batch_size
        )

        # Evaluate the model on the validation set
        val_loss, val_acc = model.evaluate(val_generator, verbose=1)
        print(f"Validation accuracy: {val_acc:.2f}")

        # Save the model
        model.save('sign_language_detection_model.keras')

    except Exception as e:
        print(f"Error during training 1: {e}")

def pred():
    try:
        model = tf.keras.models.load_model('sign_language_detection_model.keras')

        # Example image path for prediction
        img_path = 'test/M_test.jpg'

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(200, 200))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        print(f"Predicted class: {predicted_class}")

    except Exception as e:
        print(f"Error during training: {e}")

def main():
    try:
        # Define paths to your dataset
        train_dir = 'American/test1' # Replace  with file path
        val_dir = 'American/train' # Replace  with file path

        # Define image dimensions and batch size
        img_width, img_height = 200, 200
        batch_size = 256 #32

        # Use ImageDataGenerator for data augmentation and normalization
        train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True
        )

        train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode='sparse'
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        val_generator = val_datagen.flow_from_directory(
                    val_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode='sparse'
        )

        # Determine number of classes automatically from the generator
        num_classes = train_generator.num_classes

        # Build the CNN model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')  # Assuming 26 classes
        ])

        # Compile the model
        opt=Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        #Summary
        model.summary()

        print("Training started...")
        train(model, train_generator, val_generator, batch_size)
        pred()

        model_path='C:/Users/milan/Documents/PythonProjects/Nullclass/sign/sign_language_detection_model.keras'
        print("Training started...1")
        train1(model, model_path, train_generator, val_generator, batch_size)
        pred()
        print("Training started...2")
        train1(model, model_path, train_generator, val_generator, batch_size)
        pred()
        print("Training started...3")
        train1(model, model_path, train_generator, val_generator, batch_size)
        pred()
    
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == '__main__':
    main()