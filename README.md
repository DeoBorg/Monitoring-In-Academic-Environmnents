# Smart Monitoring in Academic Environments  
### Computer Vision System for Analysing Student Activity

## 📌 Overview
This project is part of a dissertation titled:

> **"Smart Monitoring in Academic Environments: Design and Implementation of a Computer Vision Tool for Analysing Student Activity"**

The system uses **computer vision techniques** to automatically analyse student behaviour in academic environments (e.g., libraries or study rooms).

It detects and classifies behaviours such as:
- 📖 Focused
- 📱 Using Phone
- 💬 Chatting
- 👀 Looking Away

---

## 🧠 System Architecture

The system is designed as a **multi-stage pipeline**, combining object detection, tracking, and behaviour inference:


---

## ⚙️ Technologies Used

- **YOLOv8 (Ultralytics)** – Object detection (person, laptop, phone)
- **ByteTrack** – Multi-object tracking (persistent person IDs)
- **OpenCV** – Video processing and visualization
- **Python** – Core implementation
- *(Planned)* **BlazePose** – Pose estimation for behaviour reasoning

---

## 📂 Project Structure
src/
│
├── person_tracking_bytetrack.py # Person tracking using YOLO + ByteTrack
├── person_object_association.py # Associates laptops/phones to tracked persons
├── person_tracking.py # Legacy tracking script
├── ByteTrack_ID_Association.py # Prototype combined pipeline
├── yolo_video_filtered.py # YOLO detection on video
├── yolo_video_save.py # YOLO detection with video output
├── test_yolo_filtered.py # YOLO testing (filtered classes)
├── test_yolo_image.py # YOLO testing on images
├── frame_extractor.py # Extract frames from videos
├── video_loader.py # Video loading utilities


---

## 🚀 Current Features

### ✅ Person Detection & Tracking
- Detects persons using YOLO
- Tracks each individual with a **persistent ID (ByteTrack)**

### ✅ Object Detection
- Detects:
  - 💻 Laptops
  - 📱 Phones

### ✅ Person–Object Association
- Assigns detected laptops and phones to the correct person
- Uses:
  - Region expansion
  - IoU + distance scoring
  - Temporal smoothing

### ✅ Video Annotation
- Displays:
  - Bounding boxes
  - Person IDs
  - Laptop/Phone association status

---

## 🧪 How to Run

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python

2. Run Person Tracking (ByteTrack)
python src/person_tracking_bytetrack.py

2. Run Person Tracking (ByteTrack)
python src/person_object_association.py

Outputs will be saved to:
outputs/annotated_videos/person_object_association.mp4