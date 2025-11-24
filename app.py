from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials
import os
from dotenv import load_dotenv
import json
from ml.predictor import Predictor
from ml.trainer import ModelTrainer
from ml.config import FIREBASE_COLLECTIONS, MODELS_DIR, METADATA_FILE

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin
cred_path = os.getenv('FIREBASE_CRED_PATH')
if not cred_path or not os.path.exists(cred_path):
    print("Wrning: Firebase credentials not found. Using application default credentials.")
    firebase_admin.initialize_app()
else:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

# Health check endpoint
@app.route('api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'API is running'}), 200

# Predict endpoint for specific category
@app.route('api/predict/<category>', methods=['POST'])
def predict_category(category):
    try:
        # Validate category
        if category not in FIREBASE_COLLECTIONS:
            return jsonify({'error': f'Invalid category: {category}'}), 400

        # Get years_ahead from request
        data = request.get_json() or {}
        years_ahead = data.get('years_ahead', 5)

        # Create predictor instance and get predictions
        predictor = Predictor(category)
        predictions = predictor.predict_future(years_ahead)

        return jsonify({
            'category': category,
            'predictions': predictions
        }), 200

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# Predict all categories
@app.route('api/predict-all', methods=['POST'])
def predict_all():
    try:
        data = request.get_json() or {}
        years_ahead = data.get('years_ahead', 5)

        results = {}
        error = {}

        for category in FIREBASE_COLLECTIONS.keys():
            try:
                predictor = Predictor(category)
                predictions = predictor.predict_future(years_ahead)
                results[category] = predictions
            except Exception as e:
                error[category] = f"Prediction failed: {str(e)}"

        response = {'predictiopns': results}
        if errors:
            response['errors'] = errors

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# Train model for specific category
@app.route('api/train/<category>', methods=['POST'])
def train_category(category):
    try:
        # Validate category
        if category not in FIREBASE_COLLECTIONS:
            return jsonify({'error': f'Invalid category: {category}'}), 400

        # Train model
        trainer = ModelTrainer(category)
        history = trainer.train(verbose=0)

        return jsonify({
            'message': f'Model trained successfully for category: {category}',
            'final_loss': float(history.history['loss'][-1]),
            'final_mae': float(history.history['mae'][-1])
        }), 200

    except Exception as e:
        return jsonify({'error': f"Training failed: {str(e)}"}), 500

# Get model information
@app.route('api/model-info/<category>', methods=['GET'])
def get_model_info(category):
    try:
        # Validate category
        if category not in FIREBASE_COLLECTIONS:
            return jsonify({'error': f'Invalid category: {category}'}), 400

        # Load metadata
        if not os.path.exists(METADATA_FILE):
            return jsonify({'error': f'Metadata file not found for category: {category}'}), 404

        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        if category not in metadata:
            return jsonify({'error': f'Model not trained for category: {category}'}), 404

        return jsonify(metadata[category]), 200

    except Exception as e:
        return jsonify({'error': f"Failed to get model information: {str(e)}"}), 500

# Get all models information 
@app.route('api/model-info-all', methods=['GET'])
def get_all_models_info():
    try:
        if not os.path.exists(METADATA_FILE):
            return jsonify({'models': {}}), 200

        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        return jsonify({'models': metadata}), 200
    
    except Exception as e:
        return jsonify({'error': f"Failed to get all models information: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Run the Flask app
    app.run(host='0.0.0.0', port=5432, debug=True)