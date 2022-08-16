import cv2
import face_recognition
cap = cv2.VideoCapture('/home/phanminhgiang/Downloads/C0027.MP4')
face_locations = []
while True:
    ret, frame = cap.read()
    rgb_fram = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_fram)
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(25) == 13:
        break