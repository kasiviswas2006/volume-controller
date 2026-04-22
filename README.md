# Finger Sensor Volume Controller

A Python-based computer vision project that allows you to control your system's master volume using hand gestures. By tracking the distance between your thumb and index finger through your webcam, you can dynamically adjust the volume in real-time.

## Features

- **Real-time Hand Tracking:** Utilizes MediaPipe to detect and track hand landmarks with high accuracy.
- **Gesture Volume Control:** Maps the distance between the thumb tip and index finger tip to the system's master volume.
- **Volume Smoothing:** Implements an exponential moving average to prevent abrupt volume changes, providing a smooth user experience.
- **Visual Feedback:** Displays the live camera feed with an intuitive UI showing the hand landmarks, a line connecting the thumb and index finger, the current distance, and the volume percentage.

## Prerequisites

This project is designed for **Windows** (due to the `pycaw` library used for system volume control).

Ensure you have Python installed (Python 3.7+ is recommended).

## Installation

1. Clone or download this repository.
2. It's recommended to create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows activate it using:
   .\venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe numpy pycaw comtypes
   ```

## Usage

1. Run the script:
   ```bash
   python volume.py
   ```
2. Your webcam will turn on. Hold your hand up in front of the camera.
3. Bring your thumb and index finger close together to lower the volume.
4. Move your thumb and index finger apart to raise the volume.
5. To exit the program, press the **`q`** key while focused on the webcam window.

## How it Works

- **OpenCV (`cv2`)**: Handles capturing the video feed from your webcam and displaying the visual UI.
- **MediaPipe (`mediapipe`)**: A machine learning framework used here specifically for its robust hand tracking model. It identifies 21 3D landmarks of your hand.
- **Math & Numpy (`math`, `numpy`)**: Used to calculate the Euclidean distance between the thumb tip (landmark 4) and index finger tip (landmark 8), and successfully interpolate this distance to the system volume range.
- **Pycaw (`pycaw`)**: Python Core Audio Windows Library, used to programmatically interface with Windows and adjust the master volume.

## Customization

You can adjust the following parameters in `volume.py` to suit your preference:
- `dist_min = 25`: The distance considered as "0% volume". Increase or decrease based on your hand size and distance from the camera.
- `dist_max = 220`: The distance considered as "100% volume".
- The smoothing factor (`0.15`) can be modified to make the volume change faster or slower based on the detected distance.
