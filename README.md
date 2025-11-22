# AI-Based Attendance System using Face Recognition

This project implements an attendance tracking system using **computer vision** and **machine learning**. It uses face recognition to identify people and logs their attendance in a CSV file.

---

## 🔍 Features

- Capture face data for different people  
- Train a K-Nearest Neighbors (KNN) classifier on the face dataset  
- Real-time face recognition using webcam  
- Voice feedback when attendance is taken  
- Attendance records stored in `.csv` files  
- Simple and portable — runs on any system with Python and a webcam

---

## 🧠 Prerequisites

- Python 3.7+  
- A webcam  
- Required Python packages:  
  ```bash
  pip install opencv-python numpy scikit-learn pywin32
  Windows only (for voice): uses SAPI.SpVoice via pywin32.

  ```
## 📂 Project Structure
  ```
     
    ├── add_faces.py                # Script to capture face images and save dataset
    ├── test.py               # Script to recognize face and mark attendance
    ├── data/
    │   ├── haarcascade_frontalface_default.xml
    │   ├── faces_data.pkl          # Pickled numpy face dataset
    │   └── names.pkl               # Pickled list of names
    ├── Attendance/                  # Folder where CSV attendance logs will be saved
    │   └── Attendance_DD-MM-YYYY.csv
    └── README.md                    # This file

```
## 🚀 How to Use

### **1. Capture face data**
Run the face data collection script:

```bash
python add_faces.py
```
Enter your name when prompted

The script will capture ~100 face images

Data will be saved in:

```
data/faces_data.pkl

data/names.pkl
```

## **2. Take attendance

Run the test script:
```
python test.py

```
The webcam will start

When your face is detected, press O to mark your attendance

Attendance will be saved into:
```
Attendance/Attendance_DD-MM-YYYY.csv
```
