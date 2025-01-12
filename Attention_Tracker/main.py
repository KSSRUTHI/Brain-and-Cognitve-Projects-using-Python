import cv2
import dlib
import numpy as np
from imutils import face_utils

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/sruthisai/PROJECT_COGNITIVE/Attention_Tracker/shape_predictor_68_face_landmarks.dat")


# Indices for eyes landmarks (based on the 68-point facial landmark model)
LEFT_EYE = slice(36, 42)
RIGHT_EYE = slice(42, 48)

def calculate_eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) to determine blink or eye movement."""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

def is_distracted(eye_aspect_ratio, threshold=0.2):
    """Determine if the user is distracted based on EAR."""
    return eye_aspect_ratio < threshold

# Open webcam feed
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye coordinates
        left_eye = shape[LEFT_EYE]
        right_eye = shape[RIGHT_EYE]

        # Calculate EAR for both eyes
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)

        # Average EAR for better stability
        ear = (left_ear + right_ear) / 2.0

        # Determine distraction
        if is_distracted(ear):
            cv2.putText(frame, "Distracted!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Focused", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw eyes on the frame
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Display the frame
    cv2.imshow("Attention Tracker", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
