import os
import pickle
import cv2
import numpy as np
import face_recognition
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from threading import Thread
import pandas as pd
from datetime import datetime
from scipy.spatial import distance as dist
import dlib

# File paths
encodings_file_path = 'face_encodings2.pkl'
attendance_file_path = 'Attendance.csv'
image_path = 'new_data'
reports_file_path = 'D:/Hackathon/old_records.csv'
patient_data_file_path = 'D:/Hackathon/patient_data.csv'  # New file for hospital patient records

# Load encodings and class names
classNames, encodeListKnown = [], []
if os.path.exists(encodings_file_path):
    with open(encodings_file_path, 'rb') as f:
        classNames, encodeListKnown = pickle.load(f)

# Load reports and patient data
reports_df = pd.read_csv(reports_file_path) if os.path.exists(reports_file_path) else pd.DataFrame(columns=["Name", "Report"])
patient_data_df = pd.read_csv(patient_data_file_path, na_values=['NA', 'nan']) if os.path.exists(patient_data_file_path) else pd.DataFrame(columns=["Name", "Details", "Image_Path", "Previous_Doctor", "Previous_Appointment"])

# Initialize dlib's face detector and shape predictor
predictor_path = 'D:/Hackathon/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# Function to find encodings
def findEncodings(images):
    return [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images if face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))]

# Process images from folder
for person_name in os.listdir(image_path):
    person_dir = os.path.join(image_path, person_name)
    if os.path.isdir(person_dir):
        images = [cv2.imread(os.path.join(person_dir, img_name)) for img_name in os.listdir(person_dir)]
        person_encodings = findEncodings(images)
        encodeListKnown.extend(person_encodings)
        classNames.extend([person_name] * len(person_encodings))

# Process images from patient data CSV
for index, row in patient_data_df.iterrows():
    if 'Image_Path' in row and isinstance(row['Image_Path'], str) and os.path.exists(row['Image_Path']):
        image = cv2.imread(row['Image_Path'])
        encodings = findEncodings([image])
        encodeListKnown.extend(encodings)
        classNames.extend([row['Name']] * len(encodings))
    else:
        print(f"Warning: Image path for {row['Name']} not found or not a valid string.")

# Save updated encodings
with open(encodings_file_path, 'wb') as f:
    pickle.dump((classNames, encodeListKnown), f)

# Attendance function
def markAttendance(name, action):
    with open(attendance_file_path, 'a+') as f:
        f.seek(0)
        if action == "login":
            if not any(name in line and "login" in line for line in f.readlines()):
                f.write(f'{name},{datetime.now().strftime("%d-%m-%y, %H:%M:%S")},login\n')
        elif action == "logout":
            if any(name in line and "login" in line for line in f.readlines()):
                f.write(f'{name},{datetime.now().strftime("%d-%m-%y, %H:%M:%S")},logout\n')

# Initialize Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
COUNTER = TOTAL = 0
liveness_detected = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def is_blinking(shape):
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    return (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0 < EYE_AR_THRESH

def is_liveness_detected(rects, gray, img):
    global COUNTER, TOTAL, liveness_detected
    liveness_detected = False
    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        if is_blinking(shape):
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                liveness_detected = True
            COUNTER = 0
    return liveness_detected

# Webcam processing
stop_flag = False
action = None
cap = None

def update_frame():
    global stop_flag, action, cap, COUNTER, TOTAL, liveness_detected

    if stop_flag:
        return

    success, img = cap.read()
    if success:
        imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        face_recognized = False
        recognized_name = ""
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                recognized_name = classNames[matchIndex].upper()
                color = (0, 255, 0)
                face_recognized = True
            else:
                recognized_name = "Unknown"
                color = (0, 0, 255)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, recognized_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        liveness_detected = is_liveness_detected(rects, gray, img)
        blink_counter_label.config(text=f"Blinks: {TOTAL}")

        liveness_text = "Liveness Detected" if liveness_detected else "Liveness Not Detected"
        liveness_color = (0, 255, 0) if liveness_detected else (0, 0, 255)
        cv2.putText(img, liveness_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, liveness_color, 2, cv2.LINE_AA)

        if face_recognized and liveness_detected and action:
            markAttendance(recognized_name, action)
            action = None
            show_patient_info(recognized_name)  # New function to handle patient info

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.imgtk = imgtk

    root.after(10, update_frame)

def start_webcam(action_type):
    global stop_flag, action, cap, TOTAL
    stop_flag = False
    action = action_type
    TOTAL = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    update_frame()

def stop_webcam():
    global stop_flag, cap
    stop_flag = True
    if cap:
        cap.release()
        cap = None

def show_patient_info(name):
    def on_register():
        Name = name_entry.get().strip().upper()
        Disease = new_patient_entry.get().strip()
        previous_doct = previous_doctor_entry.get().strip()
        previous_appt = previous_appt_entry.get().strip()

        if Name:
            # Add new patient registration to the CSV
            image_path = 'D:/Hackathon/new_data'  # Update this path with actual image saving logic
            patient_data_df.loc[len(patient_data_df)] = [Name, Disease, previous_doct, previous_appt]
            patient_data_df.to_csv(patient_data_file_path, index=False)
            tk.messagebox.showinfo("Registration", "New patient registered successfully.")
            registration_window.destroy()
        else:
            tk.messagebox.showwarning("Input Error", "Please enter a name.")

    name = name.upper()
    if name in patient_data_df["Name"].values:
        patient_info = patient_data_df[patient_data_df["Name"] == name]
        details_text = patient_info.iloc[0]["Disease"] if not patient_info.empty else "No details found."
        records_window = tk.Toplevel(root)
        records_window.title("Patient Records")
        ttk.Label(records_window, text=f"Patient: {name}", font=("Helvetica", 16)).pack(pady=10)
        ttk.Label(records_window, text=details_text, font=("Helvetica", 14)).pack(pady=10)
        ttk.Button(records_window, text="Close", command=records_window.destroy).pack(pady=10)
    else:
        registration_window = tk.Toplevel(root)
        registration_window.title("Patient Registration")
        ttk.Label(registration_window, text="New Patient Registration", font=("Helvetica", 16)).pack(pady=10)
        ttk.Label(registration_window, text="Name:", font=("Helvetica", 14)).pack(pady=5)
        name_entry = ttk.Entry(registration_window, font=("Helvetica", 14))
        name_entry.pack(pady=10)
        ttk.Label(registration_window, text="Disease:", font=("Helvetica", 14)).pack(pady=5)
        new_patient_entry = ttk.Entry(registration_window, font=("Helvetica", 14))
        new_patient_entry.pack(pady=10)
        ttk.Label(registration_window, text="Previous Doctor:", font=("Helvetica", 14)).pack(pady=5)
        previous_doctor_entry = ttk.Entry(registration_window, font=("Helvetica", 14))
        previous_doctor_entry.pack(pady=10)
        ttk.Label(registration_window, text="Previous Appointment:", font=("Helvetica", 14)).pack(pady=5)
        previous_appt_entry = ttk.Entry(registration_window, font=("Helvetica", 14))
        previous_appt_entry.pack(pady=10)
        ttk.Button(registration_window, text="Register", command=on_register).pack(pady=10)
        ttk.Button(registration_window, text="Close", command=registration_window.destroy).pack(pady=10)

# Main application window
root = tk.Tk()
root.title("Face Recognition System")

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

blink_counter_label = ttk.Label(root, text="Blinks: 0", font=("Helvetica", 14))
blink_counter_label.pack(pady=10)

checkin_button = ttk.Button(root, text="Check-in", command=lambda: start_webcam("login"))
checkin_button.pack(pady=10)

logout_button = ttk.Button(root, text="Logout", command=lambda: start_webcam("logout"))
logout_button.pack(pady=10)

stop_button = ttk.Button(root, text="Stop", command=stop_webcam)
stop_button.pack(pady=10)

root.mainloop()
