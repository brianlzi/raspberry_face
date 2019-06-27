# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# face detection webcam example
# usage: python face_detection_webcam.py 

# import necessary packages
import cvlib as cv
import cv2
import face_recognition
import pickle
# open webcam
webcam = cv2.VideoCapture("input.avi")

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
model_path = "trained_knn_model.clf"
    
    # Load a trained KNN model (if one was passed in)
with open(model_path, 'rb') as f:
    knn_clf = pickle.load(f)

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)
    print(face)
    print(confidence)


    
    face_locations = []
    try:
    # loop through detected faces
        for idx, f in enumerate(face):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            face_locations.append((f[1], f[2], f[3], f[0]))
        
        
        # (startX, startY) = f[0], f[1]
        # (endX, endY) = f[2], f[3]
        faces_encodings = face_recognition.face_encodings(frame, known_face_locations = face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= 0.8 for i in range(len(face_locations))]
        # Predict classes and remove classifications that aren't within the threshold
        for pred, rec, f in zip(knn_clf.predict(faces_encodings), are_matches, face_locations):
            if rec:
                name = pred
            else:
                name = "unknown"
            (startX, startY) = f[3], f[0]
            (endX, endY) = f[1], f[2]

    # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

            # text = "{:.2f}%".format(confidence[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write confidence percentage on top of face rectangle
            cv2.putText(frame, name, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)
            # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    except Exception:
        pass
    # display output
    cv2.imshow("Real-time face detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()        
