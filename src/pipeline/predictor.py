import tensorflow as tf
import pickle
import numpy as np
from src.utils import load_model, load_preprocessor
from src.utils import CLASS_NAMES

class Predictor:
    def __init__(self, model_path:str, preprocessor_path:str):
        self.model = load_model(model_path)
        self.preprocessor = load_preprocessor(preprocessor_path)

    def predict(self, img_path):
        '''
            Predicts the CIFAR-10 class of a single image.
        '''
        try:
            preprocessed_img = self.preprocessor.preprocess_single_image(img_path)
            prediction = self.model.predict(preprocessed_img)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = CLASS_NAMES[predicted_index]
            confidence = np.max(prediction) * 100

            return predicted_class_name, confidence
        except Exception as e:
            raise e

