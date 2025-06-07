# 🧠 Raspberry Pi Facial Recognition System

This project is a real-time facial recognition system built on a Raspberry Pi using Python, DeepFace (with FaceNet512), and OpenCV. Designed for local inference, it recognizes users by comparing live camera input to stored embeddings — all without needing an internet connection.

The goal was to create a compact, deployable face recognition system that runs fully on edge devices, while offering a lightweight user interface using the Sense HAT’s LED matrix.

## 🚀 Key Features

- 📸 **Real-time face matching** using DeepFace + FaceNet512  
- 🎛 **Two modes**: Enrollment (capture & store) and Recognition (match live faces)  
- 🔴🟢 **LED feedback system** on Sense HAT to display status (match, no match, saving)  
- ⚙️ Runs entirely on Raspberry Pi 4B with a USB webcam  
- 🧠 Uses cosine similarity to match 512-dimension embeddings  

## 🖼️ Demo

![Enrollment Mode](images/enroll_mode.jpg)  
*Enrollment mode showing joystick interaction and image capture*

![Recognition Mode]
<img src="https://github.com/user-attachments/assets/d9686174-2137-489e-a9f9-bfaea50a75b9" alt="VideoCapture_20250430-210134" width="500"/>


*Recognition mode highlighting known vs. unknown faces using overlays and LED colors*

## 🛠️ Tech Stack

- Python 3.9  
- DeepFace (FaceNet512)  
- OpenCV  
- NumPy  
- Raspberry Pi OS 64-bit  
- Sense HAT  

## 🔄 How It Works

1. **Enrollment Mode**  
   - Press joystick → Capture image → Generate face embedding → Save to disk  
   - Visual feedback via Sense HAT (blue for capture, purple for save)

2. **Live Recognition Mode**  
   - Continuously captures frames every 2 seconds  
   - Compares current frame’s embedding to known users using cosine similarity  
   - Match: green LED + name overlay | No Match: red LED + warning overlay  

