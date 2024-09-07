import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Use PIL for resizing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

base_dir = '/Users/6581/Downloads/Corrosion'
rust_dir = os.path.join(base_dir, 'rust')
no_rust_dir = os.path.join(base_dir, 'no rust')

# Get all images
rust_images = os.listdir(rust_dir)
no_rust_images = os.listdir(no_rust_dir)

# Randomly select 10 images from each class for testing
test_rust = random.sample(rust_images, 10)
test_no_rust = random.sample(no_rust_images, 10)

# Create a test set and remove these images from the training set
test_set = test_rust + test_no_rust

# Remove test images from training set
rust_images = [img for img in rust_images if img not in test_rust]
no_rust_images = [img for img in no_rust_images if img not in test_no_rust]

# Define image size and batch size
img_height, img_width = 150, 150
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    classes=['rust', 'no rust'],
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    classes=['rust', 'no rust'],
    subset='validation'
)

def create_simple_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
cnn_model = create_simple_cnn()
cnn_model.fit(train_generator, validation_data=validation_generator, epochs=10)

cnn_model.save('simple_cnn_model.h5')

# Updated image loading function using PIL for resizing
def load_test_images(test_image_paths):
    test_data = []
    labels = []
    
    # Load images from rust and no_rust folders
    for img_name in test_image_paths:
        img_path = os.path.join(rust_dir if img_name in test_rust else no_rust_dir, img_name)
        
        if os.path.exists(img_path):  # Ensure the path exists
            img = Image.open(img_path)
            img = img.resize((img_height, img_width))  # Resize using PIL
            img = np.array(img)  # Convert to numpy array
            test_data.append(img)
            
            # Assign class label
            if 'rust' in img_path:
                labels.append(1)  # Class 1 for rust
            else:
                labels.append(0)  # Class 0 for no rust
        else:
            print(f"File not found: {img_path}")  # Print missing file paths

    return np.array(test_data) / 255.0, np.array(labels)

# Load the test data and labels
test_data, test_labels = load_test_images(test_set)

# Predict on test data
predictions = cnn_model.predict(test_data)
predicted_classes = (predictions > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predicted_classes)
print(f'Test accuracy of Simple CNN: {accuracy * 100:.2f}%')
