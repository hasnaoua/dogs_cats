import tensorflow as tf
from model import build_model, save_model, early_stopping
from preprocess import create_dataset
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle dataset info
dataset_name = "anthonytherrien/dog-vs-cat"
dataset_path = "dog-vs-cat.zip"
extract_dir = "dog-vs-cat"

def download_and_extract_dataset():
    """
    Downloads the dataset from Kaggle and extracts it to a folder.
    """
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset (ensure Kaggle.json is set up correctly in ~/.kaggle/)
    print("Downloading dataset...")
    api.dataset_download_files(dataset_name, path=".", unzip=False)

    # Unzip the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Dataset extracted to:", os.path.abspath(extract_dir))


def main(epochs=100, batch_size=32):
    """
    Main function to create datasets, build the model, and train it.
    """
    # Download and extract dataset
    if not os.path.exists(extract_dir):
        download_and_extract_dataset()

    # Create datasets from the extracted folder
    train, valid = create_dataset(os.path.abspath(extract_dir + "/animals"), batch_size=batch_size)

    # Build the model
    classifier = build_model()

    # Compile the model for multi-class classification
    classifier.compile(
        optimizer='adam', 
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Train the model using the `train` and `valid` datasets
    classifier.fit(
        train, 
        epochs=epochs,  # Use the parameterized number of epochs
        validation_data=valid, 
        callbacks=[early_stopping],
        verbose=1
    )

    # save model
    save_model(classifier, "check_points/model.keras")


if __name__ == "__main__":
    main()  # Run the main function
