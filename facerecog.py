import os
import face_recognition
import cv2
import openpyxl
from datetime import datetime, timedelta

def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            known_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(known_image)[0]
            name = os.path.splitext(filename)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

    return known_face_encodings, known_face_names

def create_daily_excel_file():
    # Generate the filename based on today's date
    date_str = datetime.now().strftime("%Y-%m-%d")
    excel_file = f'attendance_{date_str}.xlsx'
    
    # If the file doesn't exist, create it with headers
    if not os.path.exists(excel_file):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet['A1'] = 'Name'
        sheet['B1'] = 'Date'
        sheet['C1'] = 'Time'
        workbook.save(excel_file)
    
    return excel_file

def recognize_faces_and_log(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)

    # Create or load the Excel file for today
    excel_file = create_daily_excel_file()

    # Dictionary to track last log time for each person
    last_log_time = {}

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                
                # Check if the name was logged within the last hour
                now = datetime.now()
                if name not in last_log_time or now - last_log_time[name] > timedelta(hours=1):
                    last_log_time[name] = now

                    # Log attendance in the Excel sheet
                    workbook = openpyxl.load_workbook(excel_file)
                    sheet = workbook.active
                    sheet.append([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
                    workbook.save(excel_file)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

folder_path = r"C:\Users\SAI MAHESH\Desktop\files\semisters\sem7\Biometrics\mini\pics"
known_face_encodings, known_face_names = load_known_faces(folder_path)
recognize_faces_and_log(known_face_encodings, known_face_names)