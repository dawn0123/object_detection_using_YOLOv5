import torch
import numpy as np
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set device to CPU or GPU
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define class names
class_names = ['person', 'car', 'motorcycle', 'bus', 'truck']

# Load image
image_path = 'image.jpg'
image = cv2.imread(image_path)

# Resize image to input size of YOLOv5 model
# input_size = model.img_size
input_size = 448
image = cv2.resize(image, dsize=(input_size, input_size),
                   interpolation=cv2.INTER_LINEAR)

# Convert image to RGB format and normalize pixel values to [0, 1]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.array(image, dtype=np.float32) / 255.0

# Convert image to PyTorch tensor and move it to device
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

# Run object detection on image
results = model(image)

# Get bounding boxes, class IDs, and confidence scores
boxes = results.xyxy[0].cpu().numpy()
class_ids = results.pred[0].cpu().numpy()[:, 5].astype(int)
confidences = results.pred[0].cpu().numpy()[:, 4]

# Draw bounding boxes and class labels on image
for box, class_id, confidence in zip(boxes, class_ids, confidences):
    if confidence > 0.5:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        class_name = class_names[class_id]
        label = f'{class_name} {confidence:.2f}'
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display image with bounding boxes and class labels
cv2.imshow('Detection', image)
cv2.waitKey(0)
