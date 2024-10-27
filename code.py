import os
import face_recognition
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe for hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Function to extract hand landmarks from an image
def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        return landmarks
    return None

# Function to calculate similarity between two sets of landmarks
def calculate_similarity(landmarks1, landmarks2):
    landmarks1 = np.array(landmarks1)
    landmarks2 = np.array(landmarks2)
    if landmarks1.shape != landmarks2.shape:
        return 0.0
    distance = np.linalg.norm(landmarks1 - landmarks2)
    max_distance = np.linalg.norm(np.ones_like(landmarks1))  # Maximum possible distance
    similarity = 1 - (distance / max_distance)
    return similarity

# Function to process an image and return the landmarks
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return None
    landmarks = extract_hand_landmarks(image)
    if landmarks:
        return landmarks
    else:
        print("No hand gesture detected in the image.")
        return None

# Function to register a new hand gesture
def register_gesture():
    image_path = input("Enter the path to the image file with your hand gesture for registration: ")
    landmarks = process_image(image_path)
    
    if landmarks:
        folder_path = r'C:\Users\SAI MAHESH\Desktop\files\semisters\sem7\Biometrics\mini\hg'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = input("Enter a name for the registered gesture: ")
        file_path = os.path.join(folder_path, file_name + '.npy')
        np.save(file_path, landmarks)
        print(f"Gesture registered successfully as {file_name}.")
    else:
        print("Failed to register the gesture. Try again.")

# Function to recognize a hand gesture
def recognize_gesture(static_landmarks):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Show your hand gesture in front of the camera. Capturing for 10 seconds...")

    start_time = time.time()
    matched = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from the webcam.")
            break

        landmarks = extract_hand_landmarks(frame)
        if landmarks:
            similarity = calculate_similarity(static_landmarks, landmarks)
            similarity_percentage = similarity * 100
            
            if similarity_percentage >= 50:
                cv2.putText(frame, "Matched", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                matched = True
                print(f"Gesture matched with {similarity_percentage:.2f}% similarity")
                break
            else:
                cv2.putText(frame, "Not Matched", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Camera Feed", frame)

        if time.time() - start_time > 10:
            break
    
    cap.release()
    cv2.destroyAllWindows()

    if not matched:
        print("No matching gesture found within 10 seconds.")

# Function to load known faces from images
def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            image_path = os.path.join(folder_path, filename)
            known_image = face_recognition.load_image_file(image_path)

            # Encode the face
            face_encoding = face_recognition.face_encodings(known_image)[0]

            # Extract the name from the filename (remove extension)
            name = os.path.splitext(filename)[0]

            # Append encoding and name to lists
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

    return known_face_encodings, known_face_names

# Function to recognize faces from the camera
def recognize_faces(known_face_encodings, known_face_names):
    # Open the default camera
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Resize frame for faster face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Compare faces in the camera feed with the known faces
        for face_encoding in face_encodings:
            # Compare the face with the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Check if any known face matches
            if True in matches:
                # Find the index of the first match
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Display the name of the person
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Main function to choose between sign-in and register
def main():
    choice = input("Choose an option (1: Sign In, 2: Register): ")
    
    if choice == '1':
        # Sign in with face and gesture recognition
        folder_path = r"C:\Users\SAI MAHESH\Desktop\files\semisters\sem6\prj\iot\smart_mirror\pics"
        known_face_encodings, known_face_names = load_known_faces(folder_path)
        
        # Recognize face
        recognize_faces(known_face_encodings, known_face_names)
        
        # Recognize gesture
        gesture_folder_path = r'C:\Users\SAI MAHESH\Desktop\files\semisters\sem7\Biometrics\mini\hg'
        gesture_file = input("Enter the name of the registered gesture file: ")
        gesture_file_path = os.path.join(gesture_folder_path, gesture_file + '.npy')
        if os.path.exists(gesture_file_path):
            static_landmarks = np.load(gesture_file_path)
            recognize_gesture(static_landmarks)
        else:
            print("Gesture file not found.")
    
    elif choice == '2':
        # Register a new gesture
        register_gesture()
    
    else:
        print("Invalid choice. Please choose 1 for Sign In or 2 for Register.")

if __name__ == "__main__":
    main()
