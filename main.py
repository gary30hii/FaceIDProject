import os

import cv2
import tensorflow as tf
from FaceID import set_face_id, resize_frame_to_square, verify
from layer import L1Dist

while True:
    # Display options to the user
    print("Options:")
    print("1. Set up Face ID")
    print("2. Real Time Test")
    print("3. Quit")

    # Get user input
    choice = input("Enter your choice: ")

    if choice == "1":
        # Logic for setting up ID
        set_face_id()
    elif choice == "2":
        # Load tensorflow/keras model
        model = tf.keras.models.load_model('/Users/gary/PycharmProjects/FaceID/siamesemodel.h5',
                                           custom_objects={'L1Dist': L1Dist})
        # OpenCV Real Time Verification
        cap = cv2.VideoCapture(0)
        print("Press 'v' to verify")
        print("Press 'q' to quit")
        while cap.isOpened():
            ret, frame = cap.read()

            # Resize the cropped square region to 250x250
            frame = resize_frame_to_square(frame, 250)

            # Show resized frame
            cv2.imshow('Verification', frame)

            # Verification trigger
            if cv2.waitKey(10) & 0xFF == ord('v'):
                # Save input image to application data/input_image folder
                cv2.imwrite(os.path.join('application_data', 'input_images', 'input_image.jpg'), frame)
                # Run verification
                results, verified = verify(model, 0.475, 0.5 )

                if verified:
                    print("verified")
                else:
                    print("not verified")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    elif choice == "3":
        # Quit the program
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please enter a valid option.")
