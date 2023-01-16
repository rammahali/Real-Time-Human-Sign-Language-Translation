import cv2
import mediapipe as mp
import numpy as np
import  os
from tensorflow.python import keras

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
actions = np.array(['hello', 'thanks', 'iloveyou'])
data_path = os.path.join('mp data')
no_sequences = 30
sequence_length = 30


def mediapipe_detection(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    result = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, result


def draw_landmark (image , result) :
    mp_drawing.draw_landmarks(image , result.face_landmarks , mp_holistic.POSE_CONNECTIONS , mp_drawing.DrawingSpec(color=(80, 110 ,10) , thickness=1 , circle_radius=1) , mp_drawing.DrawingSpec(color=(100, 256 ,21) , thickness=1 , circle_radius=0.1))
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)






def extract_landmarks(results) :
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    left_hand = np.array([[res.x, res.y, res.z] for res in
                          results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in
                           results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    return np.concatenate([pose , face , left_hand , right_hand])


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]





sequence = []
sentence = []
predictions = []
threshold = 0.5

model = keras.models.load_model(os.path.join('action_detection.h5'))

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():


        ret, frame = cap.read()


        image, results = mediapipe_detection(frame, holistic)
        print(results)


        draw_landmark(image, results)


        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))


            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]


        cv2.rectangle(image, (0, 0), (640, 40), (128, 0, 128), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('Human sign language detection', image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

