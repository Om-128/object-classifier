import os
import sys
import numpy as np
from src.pipeline.predictor import Predictor
from dataclasses import dataclass

@dataclass
class PredictPipelineConfig:
    model_path = os.path.join('artifacts', 'model.h5')
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

class PredictPipeline:
    def __init__(self, config:PredictPipelineConfig):
        self.model_path = config.model_path
        self.preprocessor_path = config.preprocessor_path
        self.predictor = Predictor(self.model_path, self.preprocessor_path)

    def run(self, img_arr:np.ndarray):
        try:
            predicted_class_name, confidence = self.predictor.predict(img_arr)
            return predicted_class_name, confidence
        except Exception as e:
            raise e


if __name__ == "__main__":
    import cv2
    config = PredictPipelineConfig()
    pipeline = PredictPipeline(config)

    img = cv2.imread("sample6.jpg")
    
    predicted_class_name, confidence = pipeline.run(img)

    

    print(f"Predicted class name {predicted_class_name}, Confidence {confidence}")
