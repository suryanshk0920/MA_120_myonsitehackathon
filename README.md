# Face Recognition System for Hospital Patient Management

Overview

This project is a face recognition system designed to manage hospital patient records. The system uses a webcam to recognize patients, display their medical history, and manage new patient registrations. The system is built using OpenCV, face_recognition, tkinter, and other relevant libraries.
Features

    Face Recognition: Identifies patients using face recognition.
    Attendance Tracking: Logs patient check-ins and check-outs with timestamps.
    Patient Record Management: Displays patient history and allows registration of new patients.
    Liveness Detection: Ensures the subject is live using blink detection.
    Real-time Webcam Feed: Displays the live video feed from the webcam with bounding boxes around detected faces.

Prerequisites

    Python 3.6 or higher
    OpenCV
    face_recognition
    dlib
    PIL (Pillow)
    numpy
    pandas
    scipy
    tkinter

Installation

    Clone the repository:

    bash

git clone https://github.com/suryanshk0920/face_recognition_system_for_ehr
cd face_recognition_system_for_ehr

Install the required libraries:

bash

    pip install -r requirements.txt

    Ensure you have the dlib's pre-trained shape predictor:
    Download shape_predictor_68_face_landmarks.dat from dlib's model zoo and extract it to the specified path in the code.

File Structure

    face_encodings2.pkl: Stores face encodings and class names.
    Attendance.csv: Logs attendance records with timestamps.
    new_data/: Directory containing images of new patients.
    old_records.csv: Contains old patient records.
    patient_data.csv: New file for hospital patient records.

Usage

    Start the Application:
    Run the main script to launch the GUI:

    bash

    python main.py

    Check-in:
        Click the "Check-in" button to start the webcam and check in a patient.
        If the patient is recognized and liveness is detected, their check-in will be logged.

    Logout:
        Click the "Logout" button to start the webcam and log out a patient.
        If the patient is recognized and liveness is detected, their logout will be logged.

    Stop Webcam:
        Click the "Stop" button to stop the webcam feed.

Code Explanation
Key Components

    Loading Encodings and Data:
    Loads face encodings, patient data, and old records from CSV files.

    Face Recognition and Attendance:
    Utilizes the face_recognition library to find and match face encodings. Logs attendance with timestamps.

    Liveness Detection:
    Uses dlib's shape predictor and eye aspect ratio (EAR) to detect blinks and ensure the subject is live.

    Patient Record Management:
    Displays patient information if the patient is recognized. Allows registration of new patients through a separate window.

    GUI:
    Built using tkinter, the GUI includes buttons for checking in, logging out, and stopping the webcam. It displays real-time webcam feed with recognized faces and liveness detection status.

Example

When the "Check-in" button is clicked, the webcam starts, and the system detects and recognizes faces. If a recognized face with liveness detection is found, the system logs the check-in and displays patient information or prompts for new patient registration.
