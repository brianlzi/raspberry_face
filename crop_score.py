import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import tifffile as tiff
from keras.models import load_model
import numpy as np
import cvlib as cv
import cv2
import pickle
import numpy as np
import time
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, save_dir=None):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    model = load_model('FaceQnet.h5')
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        
        if not os.path.isdir(os.path.join(save_dir, class_dir)):
            os.makedirs(os.path.join(save_dir, class_dir))
            
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            face, confidence = cv.detect_face(image)
            # print(face)
            # print(confidence)

            for _, f in enumerate(face):
                try:
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    rec_face = image[startY:endY, startX:endX,:]
                    rz_face = cv2.resize(rec_face, (224, 224))
                    rz_face = np.expand_dims(rz_face, axis= 0)
                    prediction = model.predict(rz_face, batch_size=1, verbose=1)
                    score = prediction[0]
                    if float(score)<0:
                        score='0'
                    elif float(score)>1:
                        score='1'
                    print(rec_face.shape)
                    print(score)
                    name = str(score[0]) + ".tif"
                    tiff.imsave(os.path.join(save_dir, class_dir, name), rec_face[:,:,::-1])
                except Exception:
                    pass
if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Crop face and record score...")
    train("/media/danpc/Data/project/data_face/train_total", save_dir = "cache")
    print("Complete!")