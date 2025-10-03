import os
import sys

from src.components.data_preprocessing import DataPreprocessingConfig, DataPreprocessing
from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

from sklearn.model_selection import train_test_split

'''
    This class is responsible for initializing and running the training pipeline.
'''

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def run_pipeline(self):
        try:
            #Data Ingestion
            ingestion = DataIngestion(config=self.data_ingestion_config)
            (x_train, y_train), (x_test, y_test) = ingestion.load_data()

            #Data Preprocessing
            preprocessor = DataPreprocessing(config=self.data_preprocessing_config)
            (x_train_scaled, y_train_scaled), (x_test_scaled, y_test_scaled) = preprocessor.preprocess(x_train, y_train, x_test, y_test)
            print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")

            # Create validation split
            x_train_scaled, x_val_scaled, y_train_scaled, y_val_scaled = train_test_split(
                x_train_scaled, y_train_scaled, test_size=0.2, random_state=42
            )


            #Model Training
            model = ModelTrainer(config=self.model_trainer_config)
            model, history = model.train_model(
                x_train_scaled, y_train_scaled,
                x_val_scaled, y_val_scaled,
                input_shape=x_train_scaled.shape[1:],
                num_classes=y_train_scaled.shape[1],
                epochs=10,
                batch_size=64
            )

            return model, history

        except Exception as e:
            raise e

if __name__ == "__main__":
    pipeline = TrainPipeline()
    model, history = pipeline.run_pipeline()
    print(model.summary())
    