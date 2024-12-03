# Snapshot Application with Classifier Options

---

## Overview

This Python application is a GUI-based tool for live video streaming and real-time image classification using OpenCV, Tkinter, and DeepFace. It includes the following features:

- **Live Video Stream**: Displays live video from the default webcam.
- **Classifier Options**:
  - **Emotion Recognition**: Detects faces and identifies dominant emotions.
  - **Cat vs Dog**: Placeholder for a module to classify between cats and dogs.
  - **Face Recognizer**: Placeholder for a face recognition module.
- **Snapshot Functionality**: Allows users to capture and save images during live video or classification processes.

---

## Features

1. **Live Video Feed**:
   - Streams live video from the webcam using OpenCV.
   - Displays the feed in a Tkinter window.

2. **Emotion Recognition**:
   - Uses DeepFace for emotion analysis.
   - Detects faces and overlays bounding boxes with emotion labels.
   - Displays on-screen instructions: 
     - Press `s` to save a snapshot.
     - Press `q` to exit.

3. **Snapshot Capture**:
   - Saves the current video frame as an image file with a timestamped filename.

4. **Future Classifier Options**:
   - Includes placeholders for **Cat vs Dog** and **Face Recognition** functionalities.

---

## Requirements

### Python Libraries:
- **Tkinter**: For GUI creation.
- **OpenCV**: For video stream handling and face detection.
- **Pillow (PIL)**: For image processing in the GUI.
- **DeepFace**: For emotion analysis.


