import json
import os
import numpy as np
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

data_dir = r'A:\object_localization\train'
annotation_file = os.path.join(data_dir, '_annotations.coco.json')

if not os.path.exists(annotation_file):
    print(f"Annotation file not found: {annotation_file}")
    exit()

# Initialize COCO API
coco = COCO(annotation_file)
categories = coco.loadCats(coco.getCatIds())
car_category = next((cat['name'] for cat in categories if 'car' in cat['name'].lower()), None)
truck_category = next((cat['name'] for cat in categories if 'truck' in cat['name'].lower()), None)

if car_category is None or truck_category is None:
    print(f"Could not find 'car' or 'truck' in the dataset.")
    exit()

car_truck_ids = coco.getCatIds(catNms=[car_category, truck_category])
print(f"Using categories: car = '{car_category}', truck = '{truck_category}'")

def load_image_and_boxes(coco, image_id, image_dir):
    image_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_dir, image_info['file_name'])
    image = cv2.imread(img_path)
    ann_ids = coco.getAnnIds(imgIds=image_info['id'], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    boxes = [ann['bbox'] for ann in anns]
    return image, boxes

def coco_bbox_to_corners(bbox):
    x_min, y_min = bbox[0], bbox[1]
    x_max, y_max = bbox[0] + bbox[2], bbox[1] + bbox[3]
    return [x_min, y_min, x_max, y_max]

def create_dataset(coco, image_dir):
    image_ids = coco.getImgIds()
    images, boxes = [], []
    for image_id in image_ids:
        image, bbox_list = load_image_and_boxes(coco, image_id, image_dir)
        bbox_list = [coco_bbox_to_corners(bbox) for bbox in bbox_list]
        images.append(image)
        boxes.append(bbox_list)
    return images, boxes

train_images, train_boxes = create_dataset(coco, data_dir)

# Correct image resizing for consistency
original_image_shape = (640, 640)  # assuming your original images are 640x640
resized_images = [cv2.resize(img, (128, 128)) for img in train_images]

# Normalize and pad bounding boxes
fixed_box_count = 5
normalized_boxes = []
for boxes in train_boxes:
    normalized = [list(map(lambda x: x / original_image_shape[0], box)) for box in boxes]  
    while len(normalized) < fixed_box_count:
        normalized.append([-1, -1, -1, -1])  # Padding with [-1,-1,-1,-1] to distinguish from valid boxes
    normalized_boxes.append(normalized[:fixed_box_count])

resized_images_np = np.array(resized_images)
normalized_boxes_np = np.array(normalized_boxes)

# Improve model architecture with BatchNorm and more filters
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),  # Add BatchNorm
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),  # Add Dropout
        layers.Dense(4 * fixed_box_count)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model((128, 128, 3))

# Use EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=3)

model.fit(
    resized_images_np,
    normalized_boxes_np.reshape(-1, 20),
    epochs=20,  # Increased epochs for better training
    batch_size=16,
    callbacks=[early_stopping]  # Early stopping
)

def visualize_predictions(model, image, true_boxes):
    resized_image = cv2.resize(image, (128, 128))
    resized_image_input = np.expand_dims(resized_image, axis=0)
    predicted_boxes = model.predict(resized_image_input)[0]
    
    original_height, original_width = image.shape[:2]
    predicted_boxes = predicted_boxes.reshape(-1, 4) * [original_width, original_height, original_width, original_height]
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    for box in true_boxes:
        x_min, y_min, x_max, y_max = box
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='green', linewidth=2, fill=False)
        plt.gca().add_patch(rect)

    for box in predicted_boxes:
        if box[0] != -1:  # Only plot valid predicted boxes
            x_min, y_min, x_max, y_max = box
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='red', linewidth=2, fill=False)
            plt.gca().add_patch(rect)

    plt.axis('off')
    plt.show()
    
# Load a new image for prediction
def load_new_image(image_path):
    image = cv2.imread(image_path)
    return image

# Predict on a new image
def predict_and_visualize(image_path, model):
    image = load_new_image(image_path)
    visualize_predictions(model, image, [])  

# Example usage:
new_image_path = r'A:\object_localization\test\thumbmitsubishi-boom-truck-159340541_jpg.rf.04f7d31ab0356a495f698c83267307fa.jpg' 
predict_and_visualize(new_image_path, model)

#image_id = 5
#example_image, example_boxes = load_image_and_boxes(coco, image_id, data_dir)
#visualize_predictions(model, example_image, example_boxes)
