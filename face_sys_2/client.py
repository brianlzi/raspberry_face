import socket
import cv2
import face_recognition
import numpy as np


HOST = "10.42.0.1"
PORT = 8082
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Cấu hình socket
s.connect((HOST, PORT)) # tiến hành kết nối đến server

video_capture = cv2.VideoCapture('rtsp://admin:@10.42.0.42/554')

# Initialize some variables
face_locations = []
process_this_frame = True

i = 0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            frame = frame[top:bottom, left:right,:]
            try:
                frame = cv2.resize(frame,(220,220))
            except:
                continue
            idata = frame.tobytes()
            print("Gui .... ")
            s.sendall(idata) # Gửi dữ liệu lên server 

# When everything is done, release the capture    
    i += 1

    if i % 4 != 0:
        process_this_frame = False
    else:
        process_this_frame = True
