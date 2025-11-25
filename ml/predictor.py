import os
import numpy as np
from tensorflow import keras
from .data_preprocessor import DataProcessor
from .config import MODELS_DIR, PREDICTION_CONFIG

class Predictor:
    def __init__(self, category):
        self.category = category
        self.data_processor = DataProcessor(category)
        self.model = None

        # Load model and scaler
        self.load_model()

    def load_model(self):
        """Load the trained model and scaler."""
        model_path = os.path.join(MODELS_DIR, f"{self.category}_model.h5")
        scaler_path = os.path.join(MODELS_DIR, f"{self.category}_scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found for category: {self.category}. Please train the model first.")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found for category: {self.category}. Please train the model first.")

        self.model = keras.models.load_model(model_path)
        self.data_processor.load_scaler(scaler_path)

    def predict_future(self, years_ahead=None):
        """Predict future values for the category."""
        if years_ahead is None:
            years_ahead = PREDICTION_CONFIG['default_years_ahead']

        if years_ahead > PREDICTION_CONFIG['max_years_ahead']:
            years_ahead = PREDICTION_CONFIG['max_years_ahead']

        # Get input sequence from latest data
        input_sequence, last_year = self.data_processor.prepare_prediction_input()

        predictions = []
        current_sequence = input_sequence.copy()

        # Iteratively predict future values
        for i in range(years_ahead):
            # Predict next year
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0])

            # Update sequence for next prediction
            # Remove first timestamp and append prediction
            new_sequence = np.concatenate([
                current_sequence[0, 1:, :],
                next_pred.reshape(1, -1)
            ], axis=0)
            current_sequence = new_sequence.reshape(1, new_sequence.shape[0], new_sequence.shape[1])

        # Convert predictions back to original scale
        predictions_array = np.array(predictions)
        predictions_denormalized = self.data_processor.inverse_transform(predictions_array)

        # Format results
        results = []
        for i, pred in enumerate(predictions_denormalized):
            year = last_year + i + 1
            prediction_dict = {
                'year': int(year),
                'predictions': {
                    feature: max(0, round(float(value))) # Ensure non-negative integers
                    for feature, value in zip(self.data_processor.feature_names, pred)
                }
            }
            results.append(prediction_dict)

        return results

    def get_model_info(self):
        """Get information about the model."""

        return {
            'category': self.category,
            'features': self.data_processor.feature_names,
            'num_features': len(self.data_processor.feature_names),
        }