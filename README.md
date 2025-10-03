# Sign Language Detection

A computer vision project that detects and classifies sign language gestures using machine learning.

## Project Overview

This project uses a webcam to capture hand gestures, processes them using MediaPipe's hand landmark detection, and classifies them using a Random Forest classifier. The system can be trained to recognize different sign language gestures.

## Features

- Real-time hand gesture detection
- Custom dataset creation
- Machine learning-based classification
- Support for multiple gesture classes

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sign_language_detection.git
   cd sign_language_detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   Or install the primary dependencies directly:
   ```
   pip install opencv-python mediapipe scikit-learn
   ```

## Project Workflow

The project consists of three main scripts that should be run in sequence:

1. **collect_imgs.py**: Captures images from your webcam for dataset creation
2. **create_dataset.py**: Processes the captured images to extract hand landmarks
3. **train_classifier.py**: Trains a Random Forest classifier on the processed data

## Usage

### 1. Collecting Images

Run the following command to start collecting images for your dataset:

```
python collect_imgs.py
```

This script will:
- Create a `data` directory with subdirectories for each class (gesture)
- Capture 100 images for each of the 3 default classes
- Wait for you to press 'q' before starting to capture images for each class

You can modify the following variables in the script:
- `number_of_classes`: Number of different gestures to capture (default: 3)
- `dataset_size`: Number of images to capture for each class (default: 100)
- `cap = cv2.VideoCapture(2)`: Change the camera index if needed (default: 2)

### 2. Creating the Dataset

After collecting images, process them to extract hand landmarks:

```
python create_dataset.py
```

This script will:
- Process all images in the `data` directory
- Extract hand landmarks using MediaPipe
- Normalize the coordinates
- Save the processed data to `data.pickle`

### 3. Training the Classifier

Train the Random Forest classifier on the processed data:

```
python train_classifier.py
```

This script will:
- Load the processed data from `data.pickle`
- Split the data into training and testing sets
- Train a Random Forest classifier
- Evaluate the model's accuracy
- Save the trained model to `model.p`

## Dependencies

- OpenCV: For image processing and camera capture
- MediaPipe: For hand detection and landmark extraction
- scikit-learn: For machine learning algorithms
- NumPy: For numerical operations

## Customization

You can customize the project by:
- Adding more gesture classes
- Adjusting the dataset size
- Tuning the RandomForest classifier parameters
- Modifying the camera settings


## Acknowledgements

- MediaPipe for the hand landmark detection
- scikit-learn for the machine learning tools
