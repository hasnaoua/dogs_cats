import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import logging
import os


# Configure logging
logging.basicConfig(level=logging.INFO)


# Create early stopping callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.05,
    patience=10,
    restore_best_weights=True
)


# Function to build the model
def build_model(num_classes=2, dropout_rate=0.5):
    """
    Builds a convolutional neural network model for image classification.

    Parameters:
    - num_classes (int): The number of output classes (default: 2).
    - dropout_rate (float): The dropout rate for regularization (default: 0.2).

    Returns:
    - tf.keras.Model: A compiled Keras model ready for training.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for multi-class classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def save_model(model: tf.keras.Model, file_path: str, overwrite: bool = True, include_optimizer: bool = True) -> None:
    """
    Saves a TensorFlow Keras model to the specified file path.

    Parameters:
    - model (tf.keras.Model): The model to save.
    - file_path (str): Path where the model will be saved. Should end with .keras or .h5 for compatibility.
    - overwrite (bool, optional): Whether to overwrite any existing model at the target location. Defaults to True.
    - include_optimizer (bool, optional): Whether to include the optimizer state in the saved model. Defaults to True.

    Returns:
    - None

    Raises:
    - ValueError: If the provided model is not a valid Keras model.
    - ValueError: If the file_path does not have a compatible extension (.keras or .h5).
    """
    # Verify if the model is a valid Keras model
    if not isinstance(model, tf.keras.Model):
        raise ValueError("The provided model is not a valid Keras model.")
    
    # Ensure the file path has a compatible extension
    if not (file_path.endswith('.keras') or file_path.endswith('.h5')):
        raise ValueError("Invalid filepath extension for saving. Use either a `.keras` extension (recommended) or a `.h5` extension.")
    
    # Create the directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory for model saving at: {directory}")

    try:
        # Save the model
        tf.keras.models.save_model(
            model,
            filepath=os.path.abspath(file_path),
            overwrite=overwrite,
            include_optimizer=include_optimizer
        )
        logging.info(f"Model saved successfully to {os.path.abspath(file_path)}")

    except Exception as e:
        logging.error(f"Error saving model: {e}")