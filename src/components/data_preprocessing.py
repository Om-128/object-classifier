import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
import pickle
from tensorflow.keras.utils import to_categorical
from src.utils import save_preprocessor

'''
    Configuration class for Data Preprocessing.
'''
@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataPreprocessing:
    def __init__(self, config:DataPreprocessingConfig):
        self.config = config
        self.num_classes = 10

    def preprocess(self, x_train, y_train, x_test, y_test):

        try:
            """
            Preprocess CIFAR-10 dataset:
            - Normalize images
            - One-hot encode labels
            - Save preprocessing config (e.g. num_classes)
            """

            #Normalize images
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0

            #One-hot encode labels
            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

            #Save the preprocessor config
            save_preprocessor(self, self.config.preprocessor_obj_file_path)

            return (x_train, y_train), (x_test, y_test)
        except Exception as e:
            raise e

    def preprocess_single_image(self, img_array):
        """
        Preprocess single image (already loaded as np.array).
        - Resize to (32,32,3) if needed
        - Normalize
        - Expand dims for model input
        """
        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1,32,32,3)
        return img_array
