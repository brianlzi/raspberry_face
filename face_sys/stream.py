import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture('rtsp://admin:@10.42.0.42/554')
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cv2.imshow('Video', frame)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
