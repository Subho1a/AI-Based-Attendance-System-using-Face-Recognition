from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(str1)

print("Starting face recognition...")

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)




LABELS = np.array(LABELS).ravel()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    predicted_name = None
    attendance = None
    exist = False    

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        predicted_name = knn.predict(resized_img)[0]

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        csv_path = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(csv_path)

        attendance = [str(predicted_name), str(timestamp)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (50,50,255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(predicted_name), (x, y - 12),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # UI Instructions
    cv2.putText(frame, "Press O: Take Attendance", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    cv2.putText(frame, "Press Q: Exit", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ( 255,0,0), 2)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    # -------------------------
    #   SAVE ATTENDANCE (O)
    # -------------------------
    if k == ord('o'):
        if predicted_name is not None:

            speak("Attendance Taken")
            time.sleep(0.1)

            if exist:
                with open(csv_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

            print(f"✔ Attendance saved for: {predicted_name}")
            print("Auto exit after attendance...")

            time.sleep(0.7)
            break
        else:
            print("⚠ No face detected — cannot save!")
            speak("No face detected")

    # -------------------------
    #   EXIT (Q)
    # -------------------------
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
