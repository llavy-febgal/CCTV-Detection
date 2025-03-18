import cv2
import face_recognition
import sqlite3
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime

model = YOLO("best.pt") 

def get_known_faces():
    """Retrieve known faces from the database."""
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM persons")
    known_faces = [(row[0], np.frombuffer(row[1], dtype=np.float64)) for row in cursor.fetchall()]
    conn.close()
    return known_faces

video_capture = cv2.VideoCapture(0) 

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    known_faces = get_known_faces()
    identified_people = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = [face_recognition.compare_faces([enc], face_encoding)[0] for _, enc in known_faces]
        name = "Unidentified"
        
        if True in matches:
            matched_index = matches.index(True)
            name = known_faces[matched_index][0]
            identified_people.append(name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"unidentified_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"Unidentified person detected! Screenshot saved: {screenshot_path}")
            
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    results = model(frame)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = box.conf.item()
            if cls == 49 and conf > 0.5: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Knife Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"weapon_detected_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"⚠️ Weapon detected! Screenshot saved: {screenshot_path}")
                
    cv2.imshow("Face & Weapon Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
