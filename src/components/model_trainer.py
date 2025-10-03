import tensorflow as tf
import pickle
import os
from tensorflow.keras import datasets, layers, models
from dataclasses import dataclass
from src.utils import save_model

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.h5')

'''
    This class is responsible for training and saving the CNN model.
'''
class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config
    
    def build_model(self, input_shape, num_classes):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))

        # Compile the model here
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model
    
    def train_model(self, x_train, y_train, x_val, y_val, input_shape, num_classes, epochs=10, batch_size=64):
        
        try:
            model = self.build_model(input_shape, num_classes)

            history = model.fit(x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size
            )

            save_model(model, self.config.model_path)
            return model, history
        except Exception as e:
            raise e

