import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Define paths
train_dir = r'C:\Users\irfan\Desktop\Project\FER 7Emotions\dataset\train'
test_dir = r'C:\Users\irfan\Desktop\Project\FER 7Emotions\dataset\test'

# Define 7 emotion classes
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
img_size = 48

def load_dataset(base_path):
    data = []
    labels = []
    for idx, cls in enumerate(classes):
        class_folder = os.path.join(base_path, cls)
        for file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                data.append(img)
                labels.append(idx)
    return np.array(data), np.array(labels)

# Load datasets
X_train, y_train = load_dataset(train_dir)
X_test, y_test = load_dataset(test_dir)

# Reshape and normalize
X_train = X_train.reshape(-1, img_size, img_size, 1) / 255.0
X_test = X_test.reshape(-1, img_size, img_size, 1) / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Save model
model.save('emotion_model_7class.h5')
print("Model saved as emotion_model_7class.h5")