import pandas as pd
import numpy as np
import os
import dataclasses
import tensorflow as tf

'''
    Configuration class for Data Ingestion.
'''
@dataclasses.dataclass
class DataIngestionConfig:
    DATA_PATH: str = "data/cifar10.npz"

'''
    Data Ingestion class to handle loading and saving of CIFAR-10 dataset. 
'''
class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config

    def load_data(self):
        # If the dataset already exists, load it
        if os.path.exists(self.config.DATA_PATH):
            with np.load(self.config.DATA_PATH) as data:
                x_train, y_train = data['x_train'], data['y_train']
                x_test, y_test = data['x_test'], data['y_test']
        # Download the dataset if it doesn't exist
        else:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

            #Save the data
            np.savez(self.config.DATA_PATH, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        
        return (x_train, y_train), (x_test, y_test)

