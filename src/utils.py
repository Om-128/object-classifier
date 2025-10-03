import os
import sys
import pickle

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