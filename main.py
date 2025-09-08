import cv2
import mediapipe as mp
import numpy as np
from eye_detect import MyClass
from datetime import datetime, timedelta
import time

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# EAR function (same logic as before)
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    ear = (A + B) / (2.0 * C)
    return ear, pts

# Eye landmark indices from MediaPipe FaceMesh (iris-refined)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # 6 points for left eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # 6 points for right eye

# Thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Counters
COUNTER = 0
TOTAL = 0
now1 = 0
d = ""

# Appliance controller
controller = MyClass()

# Start video stream
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if now1 == 0:
        now1 = datetime.now()
        d = now1 + timedelta(seconds=8)
        print(d)

    now = datetime.now()
    if d <= now:
        print(now)
        print(TOTAL)
        controller.Blink(TOTAL)  # Call appliance action
        now1 = 0
        TOTAL = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # EAR for left and right eyes
            leftEAR, leftPts = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            rightEAR, rightPts = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eyes
            for pt in leftPts + rightPts:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            # Blink detection
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    if TOTAL == 7:
                        TOTAL = 0
                COUNTER = 0

            # Display info
            cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
