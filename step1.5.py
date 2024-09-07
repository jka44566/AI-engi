import os
import random
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import ResNet50

base_dir = '/Users/6581/Downloads/Corrosion'
rust_dir = os.path.join(base_dir, 'rust')
no_rust_dir = os.path.join(base_dir, 'no rust')

# Get all images
rust_images = os.listdir(rust_dir)
no_rust_images = os.listdir(no_rust_dir)

# Filter out missing images before selecting test images
rust_images = [img for img in rust_images if os.path.exists(os.path.join(rust_dir, img))]
no_rust_images = [img for img in no_rust_images if os.path.exists(os.path.join(no_rust_dir, img))]

# Randomly select 10 images from each class for testing
test_rust = random.sample(rust_images, 10)
test_no_rust = random.sample(no_rust_images, 10)

# Combine test set
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

def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return img_array

def load_test_images(test_image_paths):
    test_data = []
    labels = []
    for img_path in test_image_paths:
        if 'rust' in img_path:
            full_path = os.path.join(rust_dir, img_path)
            labels.append(1)  # Class 1 for rust
        else:
            full_path = os.path.join(no_rust_dir, img_path)
            labels.append(0)  # Class 0 for no rust

        if os.path.exists(full_path):
            img = load_and_preprocess_image(full_path)
            test_data.append(img)
        else:
            print(f"File not found: {full_path}")
    
    return np.array(test_data), np.array(labels)

def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the ResNet50 model
resnet_model = create_resnet50_model()
resnet_model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the model in the new format as recommended
resnet_model.save('resnet50_model.keras')

# Load and preprocess test images
test_data, test_labels = load_test_images(test_set)

# Make sure there are no missing images before predicting
if len(test_data) == len(test_labels):
    # Predict on the test set
    predictions_resnet = resnet_model.predict(test_data)
    predicted_classes_resnet = (predictions_resnet > 0.5).astype(int)

    # Calculate accuracy
    accuracy_resnet = accuracy_score(test_labels, predicted_classes_resnet)
    print(f'Test accuracy of ResNet50: {accuracy_resnet * 100:.2f}%')
else:
    print(f"Mismatch in number of test images and labels. Test data size: {len(test_data)}, Test labels size: {len(test_labels)}")
