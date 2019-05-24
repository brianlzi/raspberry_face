# lib for app
import tkinter
import numpy as np
import threading
import time
from pandas import DataFrame
import copy
import matplotlib.pyplot as plt

# lib for server socket
import socket
import matplotlib.pyplot as plt
import cv2
import time

# lib for face recognition
from datetime import datetime
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


HOST = '10.42.0.1' # Thiết lập địa chỉ address
PORT = 8082 # Thiết lập post lắng nghe
IMAGE_HEIGHT = 220
IMAGE_WIDTH = 220
IMAGE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH*3

num_img = 0
record = []
record_time = []

def server_socket():
    global num_img
    # list_face = []
    def recvall(sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    
    while(True):
        print("Ready to connect on Port ",PORT, "...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # cấu hình kết nối
        s.bind((HOST, PORT)) # lắng nghe
        s.listen(1) # thiết lập tối ta 1 kết nối đồng thời
        conn, addr = s.accept() # chấp nhận kết nối và trả về thông số
        with conn:
            try:
                # in ra thông địa chỉ của client
                print('Connected by', addr)
                
                while True:
                    # Đọc nội dung client gửi đến	
                    try:
                        data = recvall(conn, IMAGE_SIZE)
                    except:
                        continue
                        
                    try:
                        arr = np.frombuffer(data, dtype ='uint8')
                    except:
                        break
                    
                    if arr.shape[0] != 0:
                        try:
                            img = arr.reshape(IMAGE_HEIGHT,IMAGE_WIDTH,3)
                            # cv2.imwrite(str(num_img)+ ".jpg", img)
                            num_img = num_img + 1
                            print(img.shape, "===========================", num_img)
                            # list_face.append(img)
                
                            # if len(list_face) == 1:
                            #     now = datetime.now()
                            #     thr_model = threading.Thread(target= model, args= (copy.deepcopy(list_face),copy.deepcopy(now),))
                            #     thr_model.start()
                            #     list_face.clear()
                            #     # num_img = 0
                        except:
                            print(img.shape[0])
                    else:
                        print("zero")
                    
                    # Và gửi nội dung về máy khách
                    #conn.sendall(b'Hello client')
                    if not data: # nếu không còn data thì dừng đọc
                        break       
            finally:
                s.close() # đóng socket

def stream():
    list_face = []
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
                
                list_face.append(frame)
                if len(list_face) == 1:
                    now = datetime.now()
                    thr_model = threading.Thread(target= model, args= (copy.deepcopy(list_face),copy.deepcopy(now),))
                    thr_model.start()
                    list_face.clear()
                
                cv2.imwrite(str(i)+ ".jpg", frame)

    # When everything is done, release the capture    
        i += 1

        if i % 2 != 0:
            process_this_frame = False
        else:
            process_this_frame = True                            

                

def model(cp_list,cp_now):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    global record
    distance_threshold = 0.5
    model_path = "trained_knn_model.clf"
    
    # Load a trained KNN model (if one was passed in)
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    for X_img in cp_list:
        # Load image file and find face locations
        X_face_locations = face_recognition.face_locations(X_img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        # Predict classes and remove classifications that aren't within the threshold
        for pred, rec in zip(knn_clf.predict(faces_encodings), are_matches):
            if rec:
                name = pred
            else:
                name = "unknown"
        if name not in record:
                end = datetime.now()
                record.append(name)
                record_time.append(end-cp_now)
  


def save_file(cp_record, cp_time):
    dic = {"Name": cp_record, "Time" : cp_time}
    df = DataFrame(dic, columns= ['Name', 'Time'])
    df.to_csv ("checking.csv", index = None, header=True) #Don't forget to add '.csv' at the end of the path
    tkinter.Label(window, text = str(df)).pack()



def start():
    thr_stream = threading.Thread(target = stream)
    thr_stream.start()

    thr_socket = threading.Thread(target = server_socket)
    thr_socket.start()
    
def call_file():
    thr_file = threading.Thread(target = save_file, args = (copy.deepcopy(record),copy.deepcopy(record_time),))
    thr_file.start()

if __name__=="__main__":
    window = tkinter.Tk()
    window.title("Face Recognition")
    window.geometry("300x200")    

    tkinter.Button(window, text = "Run!", command = start).pack() 
    tkinter.Button(window, text = "Extract file!", command = call_file).pack()
    
    window.mainloop()     
    
    
    

