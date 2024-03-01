# Facial Recognition System with Siamese Network

This repository contains an implementation of a facial recognition system utilizing a Siamese network architecture. The system offers functionalities for setting up Face ID by capturing and storing user images for model training, as well as real-time verification of user identity through a webcam.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV

## Installation

1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.

## Usage

### Set up Face ID

1. Run the script with `python main.py`.
2. Select "1. Set up Face ID" from the menu.
3. Follow the on-screen instructions to capture your images and train the model.

### Real-time verification

1. Run the script again.
2. Select "2. Real Time Test" from the menu.
3. Press 'v' on your keyboard to trigger verification when your face is in the frame.
4. The program will display "verified" or "not verified" depending on the result.
5. Press 'q' to quit.

## Note

- This implementation serves as a basic educational tool.
- Adjustments are necessary for production use, including considerations for user authentication, security, and ethical implications.

## Files

- `main.py`: Main script containing the user interface, set-up, and verification functionalities.
- `FaceID.py`: Functions for capturing images, image preprocessing, and verification logic.
- `layer.py`: Custom L1 distance layer for the Siamese network.
- `siamesemodel.h5`: The trained Siamese network model (generated during set-up).
- `data/`: Folder containing training image datasets (created during set-up).
- `application_data/`: Folder for storing captured verification images.

## Disclaimer

This code is provided for educational purposes only. Use it responsibly and be aware of the ethical and legal implications of facial recognition technology.
