import os
import shutil

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.metrics import Recall, Precision
from tensorflow.keras.layers import Dense
import uuid
import cv2

from layer import L1Dist

POS_PAS = os.path.join('data', 'positive')
NEG_PAS = os.path.join('data', 'negative')
ANC_PAS = os.path.join('data', 'anchor')


def resize_frame_to_square(frame, size):
    # Determine the smaller dimension (width or height)
    min_dim = min(frame.shape[0], frame.shape[1])

    # Calculate the size of the square region
    square_size = min_dim

    # Calculate the coordinates for cropping the center square region
    start_x = (frame.shape[1] - square_size) // 2
    start_y = (frame.shape[0] - square_size) // 2
    end_x = start_x + square_size
    end_y = start_y + square_size

    # Crop the frame to the square region
    cropped_frame = frame[start_y:end_y, start_x:end_x]

    # Resize the cropped square region to the specified size
    resized_frame = cv2.resize(cropped_frame, (size, size))

    return resized_frame


def clear_directory(directory):
    try:
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            # Construct the full path to the file
            file_path = os.path.join(directory, filename)
            try:
                # Attempt to remove the file
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                # Print an error message if the file could not be removed
                print(f"Error deleting {file_path}: {e}")
    except Exception as e:
        # Print an error message if the directory couldn't be created
        print(f"Error creating directory {directory}: {e}")


def capture_images(num_images, output_folder):
    # Open the default camera (usually webcam)
    cap = cv2.VideoCapture(0)
    cv2.waitKey(1)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the specified number of images
    for i in range(num_images):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = resize_frame_to_square(frame, 250)

        # Display the captured frame
        cv2.imshow('Capture', frame)

        # Save the captured frame
        image_path = os.path.join(output_folder, f'image_{i}.jpg')
        cv2.imwrite(image_path, frame)

        # Print progress
        print(f"Image {i+1}/{num_images} captured: {image_path}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


def capture_and_save_images(capture_path, num_images):
    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)
    counter = 0

    while cap.isOpened() and counter < num_images:
        ret, frame = cap.read()

        # Resize the cropped square region to 250x250
        frame = resize_frame_to_square(frame, 250)

        # Display the captured image
        cv2.imshow("Captured Image", frame)        # Create a unique file path
        img_name = os.path.join(capture_path, '{}.jpg'.format(uuid.uuid1()))

        # Save the image
        cv2.imwrite(img_name, frame)
        print(f"Saved image: {img_name}")

        counter += 1

    cap.release()
    cv2.destroyAllWindows()


def copy_images(source_path, destination_path):
    # Iterate over all files in the source directory
    for filename in os.listdir(source_path):
        # Construct the full paths
        source_file = os.path.join(source_path, filename)
        destination_file = os.path.join(destination_path, filename)

        # Copy the file
        shutil.copyfile(source_file, destination_file)
        print(f"Copied image from {source_file} to {destination_file}")


def capture_images_and_save():

    # Clear the anchor directory
    clear_directory(ANC_PAS)

    # Capture and save 100 images to the anchor directory
    capture_images(100, "data/anchor")

    # Clear the positive directory
    clear_directory(POS_PAS)

    # Capture and save 100 images to the positive directory
    capture_images(100, "data/positive")

    clear_directory('application_data/verification_images')

    # Copy all images from the positive directory to the verification directory
    copy_images("data/positive", 'application_data/verification_images')
    print("\n\nScanning done\n\n")


# Preprocessing - Scale and Resize
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 105 x 105 x 3
    img = tf.image.resize(img, (105, 105))
    # Scale image to be between 0 and 1
    img = img / 255.0

    return img


def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label


# Build Embedding Layer
def make_embedding():
    inp = Input(shape=(105, 105, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


# Make Siamese Model
def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(105, 105, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(105, 105, 3))

    # Combine siamese
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    embedding = make_embedding()
    distances = siamese_layer((embedding(input_image), embedding(validation_image)))

    # Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# Build Train Step Function
@tf.function
def train_step(batch, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = model(X, training=True)
        # Calculate loss
        loss = loss_fn(y, yhat)

    # Calculate gradients
    grad = tape.gradient(loss, model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    # Return loss
    return loss


# Build Training Loop
def train(data, model, loss_fn, optimizer, EPOCHS, checkpoint, checkpoint_prefix):
    # Loop through epochs
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch, model, loss_fn, optimizer)
            progbar.update(idx + 1)

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def evaluate_model(model, test_data):
    # Evaluate Model
    r = Recall()
    p = Precision()
    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat)
    print(r.result().numpy(), p.result().numpy())
    # Save Model
    # Save weight
    model.save('siamesemodel.h5')


def set_face_id():
    capture_images_and_save()
    anchor = tf.data.Dataset.list_files(ANC_PAS + r'/*.jpg').take(100)
    positive = tf.data.Dataset.list_files(POS_PAS + r'/*.jpg').take(100)
    negative = tf.data.Dataset.list_files(NEG_PAS + r'/*.jpg').take(100)
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    # Build dataloader pipeline
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)

    # Training partition
    train_data = data.take(round(len(data) * 0.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    # Testing partition
    test_data = data.skip(round(len(data) * 0.7))
    test_data = test_data.take(round(len(data) * 0.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    siamese_model = make_siamese_model()

    # Setup Loss and Optimizer
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4)

    # Establish Checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
    train(train_data, siamese_model, binary_cross_loss, opt, EPOCHS=50, checkpoint=checkpoint,
               checkpoint_prefix=checkpoint_prefix)
    evaluate_model(siamese_model, test_data)


# Verification function
def verify(model, detection_threshold, verification_threshold):
    # Build results array
    # Define the directory path
    directory = 'application_data/verification_images'

    results = []
    for image in os.listdir('application_data/verification_images'):
        input_img = preprocess(os.path.join('application_data/input_images/input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

        # Detection Threshold: Metric above which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    print(verification)
    verified = verification > verification_threshold

    return results, verified
