from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
from time import time
import sqlite3
from datetime import datetime

# ------------------- Database Setup -------------------
# Connect to SQLite DB (creates file if not exists)
conn = sqlite3.connect("emotions.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emotion TEXT,
        confidence REAL,
        timestamp TEXT
    )
''')
conn.commit()

# ------------------- Model & Haar Setup -------------------
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

classifier = load_model(r'C:\SIH\EmotionDetectionCNN\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# ------------------- Timer Setup -------------------
last_logged_time = 0  # timestamp of last DB entry
log_interval = 60     # seconds (1 minute)

# ------------------- Camera Stream -------------------
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            confidence = float(np.max(prediction))  # highest probability
            label_position = (x, y-10)

            # ---------- Log to DB once per minute ----------
            current_time = time()
            if current_time - last_logged_time >= log_interval:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "INSERT INTO emotions (emotion, confidence, timestamp) VALUES (?, ?, ?)", 
                    (label, confidence, timestamp)
                )
                conn.commit()
                last_logged_time = current_time

            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
