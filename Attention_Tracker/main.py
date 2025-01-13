import cv2
import dlib
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/sruthisai/PROJECT_COGNITIVE/Attention_Tracker/shape_predictor_68_face_landmarks.dat")

LEFT_EYE = slice(36, 42)
RIGHT_EYE = slice(42, 48)

def calculate_eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

def is_distracted(eye_aspect_ratio, threshold=0.2):
    return eye_aspect_ratio < threshold

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
      
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[LEFT_EYE]
        right_eye = shape[RIGHT_EYE]

        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if is_distracted(ear):
            cv2.putText(frame, "Distracted!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Focused", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Attention Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
