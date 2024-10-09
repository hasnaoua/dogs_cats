from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_dataset(files_path, batch_size=32, target_size=(256, 256)):
    """
    Creates training and validation datasets from an image directory.

    Parameters:
    - files_path (str): The path to the directory containing the images.
    - batch_size (int): The number of samples to process in each batch (default: 32).
    - target_size (tuple): The target size to resize the images (default: (256, 256)).

    Returns:
    - tuple: A tuple containing the training and validation datasets.
    """
    # Check if the specified path exists
    if not os.path.exists(files_path):
        raise ValueError(f"The specified path '{files_path}' does not exist.")

    datagen = ImageDataGenerator(rescale=1./255, 
                                 rotation_range=90,
                                 width_shift_range=0.1, 
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 validation_split=0.2)

    train_dataset = datagen.flow_from_directory(
        files_path,
        seed=123, 
        subset="training",
        batch_size=batch_size,
        target_size=target_size,
        shuffle=True,
        class_mode='binary'  # Set to categorical for multi-class classification
    )

    valid_dataset = datagen.flow_from_directory(
        files_path,
        seed=123, 
        subset="validation",
        batch_size=batch_size,
        target_size=target_size,
        shuffle=True,
        class_mode='binary'  # Set to categorical for multi-class classification
    )

    return train_dataset, valid_dataset
