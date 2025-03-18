from ultralytics import YOLO
import cv2
import face_recognition
import numpy as np
import sqlite3
import smtplib
import os
from email.message import EmailMessage

EMAIL_SENDER = "febgal6@gmail.com" 
EMAIL_PASSWORD = "zmxa mwrk ceiu eaka"  
EMAIL_RECEIVER = "estellavyukusenge06@gmail.com" 

def send_email_alert(image_path, person_name, weapon_name):
    msg = EmailMessage()
    msg["Subject"] = f"⚠️ Weapon Detected: {weapon_name}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content(f"A {weapon_name} was detected! The person identified is: {person_name}.")

    with open(image_path, "rb") as f:
        image_data = f.read()
        msg.add_attachment(image_data, maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
    
    print(f"📩 Email alert sent to {EMAIL_RECEIVER}!")

weapon_model = YOLO("best.pt")

def get_known_faces():
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM persons")
    
    known_faces = []
    for row in cursor.fetchall():
        name, encoding_blob = row
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        known_faces.append((name, encoding))

    conn.close()
    return known_faces

known_faces = get_known_faces()

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    person_name = "Unidentified"

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = [face_recognition.compare_faces([enc], face_encoding)[0] for _, enc in known_faces]
        
        if True in matches:
            matched_index = matches.index(True)
            person_name = known_faces[matched_index][0]

        color = (0, 255, 0) if person_name != "Unidentified" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, person_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    results = weapon_model(frame)
    weapon_detected = False
    detected_weapon_name = ""

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            class_id = int(class_id)
            weapon_name = weapon_model.names[class_id]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{weapon_name} {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if weapon_name in ["gun", "knife"]:  
                weapon_detected = True
                detected_weapon_name = weapon_name

    if weapon_detected:
        screenshot_filename = f"weapon_detected_{person_name}.jpg"
        cv2.imwrite(screenshot_filename, frame)
        print(f"⚠️ ALERT: {detected_weapon_name} detected! ({person_name}) - Screenshot saved.")
        
        send_email_alert(screenshot_filename, person_name, detected_weapon_name)

    cv2.imshow("Face & Weapon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
