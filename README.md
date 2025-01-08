![can2](https://github.com/user-attachments/assets/f23619f7-20fb-49d6-b485-5f731c7e3d5b)
![can5](https://github.com/user-attachments/assets/2e4e2aca-a996-4079-8faf-109fb3affdd1)
![can6](https://github.com/user-attachments/assets/25a71d1e-37f7-4fff-b2ea-ab6f491a888a)
# Canc![can1](https://github.com/user-attachments/assets/1feaa6ec-cf6c-42b8-9fda-409d3405d822)

# **Description:**
This project implements YOLOv9 for cancer detection, focusing on distinguishing between benign and malignant cases in medical images. The model is trained using labeled datasets for tumor classification, aiming to assist in early diagnosis and analysis.

#**YOLOv9 Model:**
OLOv9 is an advanced version of the YOLO (You Only Look Once) model, known for its speed and accuracy in object detection tasks. It performs real-time object detection and is particularly efficient when applied to medical image classification tasks, such as detecting cancerous growths.

#**OpenCV**
OpenCV (Open Source Computer Vision Library) is used for image processing tasks in this project. It helps in reading, displaying, and manipulating the images used for detection. OpenCV integrates seamlessly with YOLOv9, enabling effective and real-time detection.

#**Installing YOLOv9 and Dependencies**
**Clone the YOLOv9 repository:**
git clone https://github.com/your-username/yolov9.git
pip install -r requirements.txt

#**Model:**
Download trained model trainable weights to load for real time testing with file name best.py.

#**Dataset.**
You can download and check dataset from

#**How to Run the Code**
Clone this repository.
Install the required dependencies.
Run the TESTING.py file

Download the requirements.txt, replace the image path with your input image file and the YOLO model path with your trained model in the main.py code, then execute the script to detect cancerous regions and classify them as Benign or Malignant.

#**Example Output:**
**Benign** (Confidence: 0.92)
**Malignant** (Confidence: 0.85)
