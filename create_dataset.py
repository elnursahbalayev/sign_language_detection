import os
import pickle
# Import mediapipe library for hand keypoint detection
import mediapipe as mp
# Import OpenCV library for image processing
import cv2
# Import matplotlib library for plotting and display
import matplotlib.pyplot as plt

# Initialize mediapipe's hand solution
mp_hands = mp.solutions.hands
# Initialize mediapipe's drawing utilities
mp_drawing = mp.solutions.drawing_utils
# Initialize mediapipe's drawing styles
mp_drawing_styles = mp.solutions.drawing_styles

# Create a Hands object for hand detection in static images
# Parameter static_image_mode is set to True, indicating the input is a static image
# Parameter min_detection_confidence is set to 0.3, meaning results are only returned if detection confidence exceeds 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data':data, 'labels':labels}, f)
f.close()

