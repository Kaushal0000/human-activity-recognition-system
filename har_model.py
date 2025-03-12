import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import sys
import signal
import tensorflow as tf
print(tf.__version__)



# Signal handler to clean up resources on exit
def signal_handler(sig, frame):
    print('Exiting...')
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Load and preprocess data
def load_data():
    # Load the training data from CSV
    train_data = pd.read_csv('Training_set.csv')
    
    # Define lists to hold image data and labels
    X_train = []
    y_train = []

    # Loop through the CSV file to load images
    for index, row in train_data.iterrows():
        img_path = os.path.join('C:\\Users\\91748\\Downloads\\archive (11)\\Human Action Recognition\\train', row['filename'])
        image = cv2.imread(img_path)
        
        if image is not None:  # Check if the image was loaded successfully
            image = cv2.resize(image, (64, 64))  # Resize the image if needed
            X_train.append(image)
            y_train.append(row['label'])
        else:
            print(f"Warning: Unable to load image {img_path}")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")

    # Normalize the pixel values
    X_train = X_train.astype('float32') / 255.0

    # Convert labels to categorical
    label_mapping = {label: idx for idx, label in enumerate(set(y_train))}
    y_train = np.array([label_mapping[label] for label in y_train])
    y_train_categorical = to_categorical(y_train)

    return X_train, y_train_categorical, len(label_mapping)

# Build the model
def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Train the model
def train_model(model, X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Make predictions using live camera feed
def predict_live(model):
    # Define the labels for the activities
    labels = ['laughing', 'fighting', 'sitting', 'cycling', 'calling', 
              'drinking', 'clapping', 'texting', 'sleeping', 'running', 
              'dancing', 'listening_to_music', 'hugging', 'using_laptop', 'eating']

    # Open the camera
    cap = cv2.VideoCapture(0)

    def preprocess_frame(frame):
        frame_resized = cv2.resize(frame, (64, 64))
        frame_normalized = frame_resized.astype('float32') / 255.0
        return np.expand_dims(frame_normalized, axis=0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = preprocess_frame(frame)

        # Predict the activity
        predictions = model.predict(input_frame)
        predicted_label = labels[np.argmax(predictions)]

        # Display the result on the frame
        cv2.putText(frame, predicted_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with the activity label
        cv2.imshow('Human Activity Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    X_train, y_train, num_classes = load_data()  # Load and preprocess your data
    model = build_model(num_classes)  # Build the model
    train_model(model, X_train, y_train)  # Train the model
    model.save('har_model.h5')  # Save the model
    predict_live(model)  # Start live prediction



#---------------------comparison of cnn knn svm random forest-----------------------------


