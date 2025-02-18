# Malory - AI-Powered Cat Deterrent System

## Overview
Malory is an advanced AI-powered deterrent system designed to prevent unwanted feline behaviors, such as excessive meowing, scratching doors, and attempting to open door handles. Leveraging machine learning, real-time signal processing, and embedded hardware, Malory detects these behaviors and triggers an automated deterrent response.

## Features
- **Audio Classification:** Detects meowing and scratching sounds using a trained neural network.
- **Computer Vision Detection:** Identifies when a cat is attempting to open a door handle.
- **Real-Time Processing:** Runs efficiently on a Raspberry Pi for live detection and response.
- **Deterrent Mechanism:** Plays a hissing sound, leveraging a cat’s natural aversion to snake-like noises.
- **Edge Deployment:** Optimized for on-device AI inference, ensuring low latency and high reliability.

## Technology Stack
### Machine Learning
- **TensorFlow / PyTorch:** Used to train deep learning models for audio classification and object detection.
- **YAMNet:** Pre-trained deep learning model for audio event classification, used for detecting meows and scratching sounds.
- **MobileNetV2:** Lightweight convolutional neural network (CNN) used for efficient computer vision tasks.
- **Librosa:** Extracts features from audio data for sound classification.
- **OpenCV:** Processes images for real-time detection of standing behavior.

### Embedded Systems
- **Raspberry Pi 4 Model B:** Handles real-time inference and deterrent mechanisms.
- **Raspberry Pi Camera Module:** Captures live video for computer vision processing.
- **USB Microphone:** Records real-time audio for sound classification.

### Software & Deployment
- **Python:** Primary programming language for data processing and AI model deployment.
- **NumPy & Pandas:** Used for dataset preprocessing and feature extraction.
- **Flask (Optional):** Provides a lightweight web interface for monitoring detection events.

## System Architecture
1. **Audio Detection Pipeline:**
   - Records incoming audio using a USB microphone.
   - Extracts MFCC (Mel-frequency cepstral coefficients) features from the audio signal.
   - Uses **YAMNet** for classifying sounds as either background noise, meowing, or scratching.
   - Triggers deterrent response if meowing or scratching is detected.

2. **Computer Vision Pipeline:**
   - Captures real-time video feed using the Raspberry Pi camera module.
   - Runs a Convolutional Neural Network (CNN) based on **MobileNetV2** to classify standing vs. non-standing behavior.
   - Triggers deterrent response if standing behavior is detected.

3. **Deterrent Response System:**
   - Activates a speaker to play a synthetic hissing sound when unwanted behavior is detected.
   - Ensures low-latency processing to provide immediate feedback.

## Model Training
The AI models were trained using a combination of real-world and publicly available datasets. 
- **Audio Model:** Trained on labeled audio samples of cat meows, scratching sounds, and background noise using **YAMNet**.
- **Vision Model:** Trained using images labeled as "Standing" and "Not Standing" to distinguish relevant behavior using **MobileNetV2**.
- **Optimization:** Models were converted to TensorFlow Lite for efficient edge inference on the Raspberry Pi.

## Hardware Requirements
| Component               | Purpose                                      |
|-------------------------|----------------------------------------------|
| Raspberry Pi 4 Model B | Main processing unit                         |
| Raspberry Pi Camera    | Captures live video for detection            |
| USB Microphone        | Captures audio for meow and scratch detection |
| Speaker               | Plays deterrent sounds                        |

## Installation & Deployment
### Prerequisites
- **Raspberry Pi OS (Latest Version)** installed on the Raspberry Pi.
- Python 3.10 or later.
- Required dependencies installed:
  
```sh
pip install numpy librosa sounddevice opencv-python tensorflow tflite-runtime
