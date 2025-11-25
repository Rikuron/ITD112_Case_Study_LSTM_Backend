import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from firebase_admin import firestore
from .config import FIREBASE_COLLECTIONS, PREDICTION_CONFIG

class DataProcessor:
    def __init__(self, category):
        self.category = category
        self.collection_path = FIREBASE_COLLECTIONS.get(category)
        self.scaler = MinMaxScaler()
        self.feature_names = []

        if not self.collection_path:
            raise ValueError(f"Invalid category: {category}")

    def fetch_data_from_firebase(self):
        """Fetch historical data from Firebase for the category."""
        db = firestore.client()

        # Navigate to the collection
        parts = self.collection_path.split('/')
        collection_ref = db.collection(parts[0]).document(parts[1]).collection(parts[2])

        docs = collection_ref.stream()

        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['Year'] = int(doc.id)
            data.append(doc_data)

        if not data:
            raise ValueError(f"No data found for category: {self.category}")

        # Convert to DataFrame and sort by year
        df = pd.DataFrame(data)
        df = df.sort_values(by='Year')

        return df

    def preprocess_data(self, df):
        """Preprocess the data for LSTM training."""
        # Extract years and features
        years = df['Year'].values
        df_features = df.drop('Year', axis=1)

        # Store feature names
        self.feature_names = df_features.columns.tolist()

        # Convert to numpy arrays
        data = df_features.values

        # Check minimum data requirement
        if len(data) < PREDICTION_CONFIG['min_sequence_length']:
            raise ValueError(
                f"Insufficient data for category: {self.category}."
                f"Minimum sequence length required: {PREDICTION_CONFIG['min_sequence_length']}"
                f"Available data: {len(data)} years"
            )
        
        # Normalize the data
        data_normalized = self.scaler.fit_transform(data)

        return years, data_normalized

    def create_sequences(self, data, sequence_length=None):
        """Create sequences for LSTM training."""
        if sequence_length is None:
            # Use all data as one sequence (for yearly predictions)
            sequence_length = len(data) - 1

        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])

        return np.array(X), np.array(y)

    def prepare_training_data(self):
        """Complete pipeline: fetch, preprocess, create sequences."""
        df = self.fetch_data_from_firebase()
        years, data_normalized = self.preprocess_data(df)

        # Dynamically determine sequence
        total_years = len(data_normalized)

        if total_years < 10:
            sequence_length = max(3, int(total_years * 0.6))
        elif total_years < 20:
            sequence_length = int(total_years * 0.6)
        else:
            sequence_length = min(15, int(total_years * 0.5))

        X, y = self.create_sequences(data_normalized, sequence_length)

        return X, y, years

    def inverse_transform(self, normalized_data):
        """Inverse transform the normalized data to the original scale."""
        return self.scaler.inverse_transform(normalized_data)

    def save_scaler(self, filepath):
        """Save the scaler to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)

    def load_scaler(self, filepath):
        """Load the scaler from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']

    def prepare_prediction_input(self, last_n_years=None):
        """Prepare input data for prediction."""
        df = self.fetch_data_from_firebase()
        years, data_normalized = self.preprocess_data(df)

        if last_n_years is None:
            # Use all available data
            input_sequence = data_normalized
        else:
            # Use last N years
            input_sequence = data_normalized[-last_n_years:]

        # Reshape for LSTM input
        input_sequence = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])

        return input_sequence, years[-1]
