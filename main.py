import tensorflow as tf
from model import build_model, early_stopping
from preprocess import create_dataset
import kagglehub

# Path to dogs and cats images
try:
    PATH = kagglehub.dataset_download("anthonytherrien/dog-vs-cat")
except Exception as e:
    print(f"An error occurred while downloading the dataset: {e}")
    exit()

def main(epochs=100, batch_size=32):
    """
    Main function to create datasets, build the model, and train it.
    """
    # Create datasets
    train, valid = create_dataset(PATH, batch_size=batch_size)

    # Build model
    classifier = build_model()

    # Compile the model for multi-class classification
    classifier.compile(optimizer='adam', 
                       loss=tf.keras.losses.CategoricalCrossentropy(),
                       metrics=['accuracy'])

    # Train the model using the `train` and `valid` datasets
    classifier.fit(
        train, 
        epochs=epochs,  # Use the parameterized number of epochs
        validation_data=valid, 
        callbacks=[early_stopping],
        verbose=1
    )

if __name__ == "__main__":
    main()  # Run the main function
