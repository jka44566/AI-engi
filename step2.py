import os
import numpy as np
import cv2
from Mask_RCNN.mrcnn import model as modellib, utils
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn import config
from tensorflow.keras import backend as KE


class YourDatasetConfig(config):
    """Configuration for training on your dataset."""
    NAME = "your_dataset"
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Replace 5 with the actual number of classes in your dataset
    
    # Other configuration parameters
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BACKBONE = "resnet101"
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # Add other parameters as needed

# Initialize the model with your configuration
config = YourDatasetConfig()
model = modellib.MaskRCNN(mode="training", model_dir=os.getcwd(), config=config)

# Initialize the model with your configuration
config = YourDatasetConfig()
model = modellib.MaskRCNN(mode="training", model_dir=os.getcwd(), config=config)
# Paths to your files
MODEL_PATH = 'path_to_trained_weights.h5'
TEST_IMAGES_DIR = 'path_to_test_images'
OUTPUT_DIR = 'path_to_output_images'

# Load the trained Mask R-CNN model
config = YourDatasetConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=os.getcwd(), config=config)
model.load_weights(MODEL_PATH, by_name=True)

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to count detected logs and save images
def process_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.detect([image], verbose=1)
    r = results[0]
    
    # Count detected logs
    num_logs = len(r['rois'])  # Number of detected objects
    print(f"Number of logs detected in {os.path.basename(image_path)}: {num_logs}")
    
    # Draw the bounding boxes and masks
    vis_image = visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], 
        config.CLASS_NAMES, r['scores'], title="Detected Logs"
    )
    
    # Save the output image
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, vis_image)

# Process all images in the test directory
for file_name in os.listdir(TEST_IMAGES_DIR):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(TEST_IMAGES_DIR, file_name)
        process_image(image_path)

print("Processing completed.")
