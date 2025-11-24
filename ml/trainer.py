import os
import json
from datetime import datetime
from .lstm_model import LSTMModel
from .data_preprocessor import DataProcessor
from .config import MODELS_DIR, METADATA_FILE

class ModelTrainer:
    def __init__(self, category):
        self.category = category
        self.data_processor = DataProcessor(category)
        self.model = None
        self.history = None

        # Ensure models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)

    def train(self, verbose=1):
        """Train the model for the category."""
        print(f"Training model for category: {self.category}")

        # Prepare data
        print("Fetching and preprocessing data...")
        X_train, y_train, years = self.data_processor.prepare_training_data()

        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Years range: {years[0]} - {years[-1]}")

        # Build model
        input_shape = X_train.shape[1, X_train.shape[2]]
        output_shape = y_train.shape[1]

        self.model = LSTMModel(input_shape, output_shape)

        # Train model
        print("Training model...")
        self.history = self.model.train(X_train, y_train, verbose=verbose)

        # Save model and scaler
        self.save_model()

        # Update metadata
        self.update_metadata(X_train, y_train, years)

        print(f"Model training completed for category: {self.category} successfully!")

        return self.history

    def save_model(self):
        """Save the trained model and scaler."""
        model_path = os.path.join(MODELS_DIR, f"{self.category}_model.h5")
        scaler_path = os.path.join(MODELS_DIR, f"{self.category}_scaler.pkl")

        self.model.save(model_path)
        self.data_processor.save_scaler(scaler_path)

        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

    def update_metadata(self, X_train, y_train, years):
        """Update the metadata file with training details."""
        # Load existing metadata or create new one
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Get final metrics
        final_loss = float(self.history.history['loss'][-1])
        final_mae = float(self.history.history['mae'][-1])

        # Update metadata for this category
        metadata[self.category] = {
            'last_trained': datetime.now().isoformat(),
            'training_years': f"{years[0]} - {years[-1]}",
            'num_years': len(years),
            'num_samples': int(X_train.shape[0]),
            'num_features': int(X_train.shape[2]),
            'final_loss': final_loss,
            'final_mae': final_mae,
            'feature_names': self.data_processor.feature_names
        }

        # Save metadata
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata updated for category: {self.category}")
        print(f"Final metrics: Loss={final_loss:.4f}, MAE={final_mae:.4f}")
        print(f"Metadata saved to: {METADATA_FILE}")
