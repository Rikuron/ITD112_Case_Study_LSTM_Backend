# Machine Learning Configuration and Hyperparameters

# LSTM Model Configuration
LSTM_CONFIG = {
    'units_layer1': 64,
    'units_layer2': 32,
    'dropout_rate': 0.2,
    'epochs': 100,
    'batch_size': 8,
    'validation_split': 0.2,
    'patience': 15,
    'learning_rate': 0.001
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'default_years_ahead': 5,
    'max_years_ahead': 10,
    'min_sequence_length': 5
}

# Firebase Collections Mapping
FIREBASE_COLLECTIONS = {
    'age': 'emigrantData/age/years',
    'civil_status': 'emigrantData/civilStatus/years',
    'destination': 'emigrantData/majorDestination/years',
    'education': 'emigrantData/education/years',
    'occupation': 'emigrantData/occupation/years',
    'sex': 'emigrantData/sex/years',
    'origin': 'emigrantData/region/years',
}

# Model Storage
MODELS_DIR = 'models'
METADATA_FILE = 'models/metadata.json'