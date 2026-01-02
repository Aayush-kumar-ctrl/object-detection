# Real-Time Object Detection using YOLOv8

A real-time computer vision application that detects and labels objects from a live webcam feed using the YOLOv8 deep learning model. The project is optimized to run on CPU-only systems.

---

## 🚀 Features
- Real-time object detection using webcam
- Detects 80 common object classes (COCO dataset)
- Displays bounding boxes with class labels and confidence scores
- Runs smoothly on low-end systems (no GPU required)
- Simple and beginner-friendly implementation

---

## 🧠 Objects Detected
The model is trained on the **COCO dataset**, which includes objects such as:
- Person
- Car, Bus, Truck, Motorcycle
- Laptop, Mobile Phone, Keyboard
- Bottle, Cup, Chair
- Dog, Cat, Bird  
…and many more (80 classes in total)

> ⚠️ Objects not included in COCO (e.g., scissors) cannot be detected without custom training.

---

## 🛠 Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- COCO Pre-trained Model

---

## ⚙️ System Requirements
- Python 3.8 or higher
- Webcam
- CPU-only system (tested on Intel i3 10th Gen)
- Windows / Linux / macOS

---

## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install ultralytics opencv-python
