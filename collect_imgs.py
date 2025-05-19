import os
import cv2

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(2)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting data for class {j}')

    done = False