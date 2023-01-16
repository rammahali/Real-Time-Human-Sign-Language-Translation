import os

import cv2
import numpy as np

import main

mp_holistic =main.mp_holistic
mp_drawing = main.mp_drawing
actions =  main.actions
data_path =  main.data_path
no_sequences = 30
sequence_length = 30

def create_folders() :
    data_path = os.path.join('mp data')
    no_sequences = 30
    for action in actions:
        for sequence in range(no_sequences):
           try:
               os.makedirs(os.path.join(data_path, action, str(sequence)))
           except:
               pass


create_folders()


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(0)
    for action in actions :
        for secquence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = main.mediapipe_detection(frame, holistic)
                main.draw_landmark(image, results)
                if frame_num ==0:
                    cv2.putText(image , "STARTING COLLECTION" ,(120,200), cv2.FONT_HERSHEY_SIMPLEX , 1,(0,255,00),  4,cv2.LINE_AA)
                    cv2.putText(image, "Collecting frames for {} video number {}".format(action , secquence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, "Collecting frames for {} video number {}".format(action , secquence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                cv2.LINE_AA)
                    cv2.imshow("Add a new sign dataset", image)

                keypoints = main.extract_landmarks(results)
                npy_path = os.path.join(data_path ,action ,str(secquence) , str(frame_num))
                np.save(npy_path,keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                     break