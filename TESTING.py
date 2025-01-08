import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")  # Suppress torch warnings

sys.path.append(r"C:\Users\3s\Downloads\New folder6\yolov9")
sys.path.append(r"C:\Users\3s\Downloads\New folder6\yolov9\utils")  # Add utils folder path

from yolov9.models.experimental import attempt_load
from yolov9.utils.general import non_max_suppression
from yolov9.utils.torch_utils import select_device

import torch
from PIL import Image
import cv2
import numpy as np

# Define class names for the dataset
class_names = ["Benign", "Malignant"]  # Assuming class 0 = Benign, class 1 = Malignant

# Custom scale_coords function
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    gain = min(img0_shape[0] / img1_shape[0], img0_shape[1] / img1_shape[1])
    pad = (img0_shape[1] - img1_shape[1] * gain) / 2, (img0_shape[0] - img1_shape[0] * gain) / 2  # padding
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords /= gain  # scale
    return coords

# Custom plot_one_box function with dark color
def plot_one_box(xyxy, img, color=(0, 0, 255), label=None, line_thickness=3):  # Dark red color
    # Draw a rectangle on the image
    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, line_thickness)
    if label:
        # Draw the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), font, 0.5, color, 1)

# Load model
device = select_device('cpu')  # Use CPU instead of GPU
weights_path = r"C:\Users\3s\Downloads\New folder6\yolov9\best.pt"  # Path to your trained best.pt
model = attempt_load(weights_path)  # Removed map_location=device

# Load and preprocess image
image_path = r"C:\Users\3s\Downloads\TEST IMAGES\CANCER\1.jpg"  # Path to your test image
image = Image.open(image_path)
image = np.array(image)
image = cv2.resize(image, (640, 640))
image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

# Inference
pred = model(image_tensor, augment=False)[0]
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

# Visualize results
for det in pred:
    if det is not None and len(det):
        det[:, :4] = scale_coords(image_tensor.shape[2:], det[:, :4], image.shape).round()
        for *xyxy, conf, cls in det:
            # Map the class index to class name (Benign/Malignant)
            label = f"{class_names[int(cls)]} {conf:.2f}"
            plot_one_box(xyxy, image, label=label, color=(0, 0, 255), line_thickness=2)  # Dark red color

# Display the image with detections
cv2.imshow("Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
cv2.waitKey(0)
cv2.destroyAllWindows()
