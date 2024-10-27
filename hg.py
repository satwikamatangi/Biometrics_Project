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

# Function to capture and compare hand gestures from the camera
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

# Main function to get user input and process the image
def main():
    image_path = input("Enter the path to the image file with your hand gesture: ")
    static_landmarks = process_image(image_path)
    
    if static_landmarks:
        print("Camera will now start. Show your hand gesture.")
        recognize_gesture(static_landmarks)

if __name__ == "__main__":
    main()
