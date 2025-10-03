import os
import sys
import pickle
import tensorflow as tf

CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

def save_preprocessor(preprocessor, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(preprocessor, f)
    except Exception as e:
        raise e

def save_model(model, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)  # For Keras models
    except Exception as e:
        raise e

def load_model(path):
    return tf.keras.models.load_model(path)

def load_preprocessor(path):
    with open(path, "rb") as f:
        return pickle.load(f)