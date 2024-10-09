import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Function to build the model
def build_model(num_classes=2, dropout_rate=0.2):
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
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create early stopping callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.05,
    patience=10,
    restore_best_weights=True
)
