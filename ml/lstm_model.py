import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .config import LSTM_CONFIG

class LSTMModel:
    def __init__(self, input_shape, output_shape):
        """
        Initialize the LSTM model.

        Args:
            input_shape: Tuple of (sequence_length, num_features)
            output_shape: Number of output features (same as num_features for multi-variate)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        self.build_model()

    # CONTAINS PARAMETERS; TUNE LATER
    def build_model(self):
        """Build the LSTM architecture."""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                LSTM_CONFIG['units_layer_1'],
                activation='tanh',
                return_sequences=True,
                input_shape=self.input_shape
            ),
            layers.Dropout(LSTM_CONFIG['dropout_rate']),

            # Second LSTM layer
            layers.LSTM(
                LSTM_CONFIG['units_layer_2'],
                activation='tanh',
                return_sequences=False,
            ),
            layers.Dropout(LSTM_CONFIG['dropout_rate']),

            # Output layer
            layers.Dense(self.output_shape, activation='linear')
        ])

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=LSTM_CONFIG['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model

    def train(self, X_train, y_train, verbose=1):
        """Train the LSTM model."""
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=LSTM_CONFIG['patience'],
            restore_best_weights=True,
            verbose=verbose
        )

        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=LSTM_CONFIG['epochs'],
            batch_size=LSTM_CONFIG['batch_size'],
            validation_split=LSTM_CONFIG['validation_split'],
            callbacks=[early_stopping],
            verbose=verbose
        )

        return history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def save(self, filepath):
        """Save the model to a file."""
        self.model.save(filepath)

    def load(self, filepath):
        """Load the model from a file."""
        self.model = keras.models.load_model(filepath)

    def get_summary(self):
        """Get the model summary."""
        return self.model.summary()