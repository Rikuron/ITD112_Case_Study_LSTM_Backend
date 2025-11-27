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
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "https://itd-112-case-study.vercel.app"
])

# Initialize Firebase Admin with flexible credential handling
def initialize_firebase():
    """Initialize Firebase with support for both local and hosting environments."""
    
    # Try Application Default Credentials first (for GCP hosting)
    if os.getenv('GAE_ENV', '').startswith('standard') or os.getenv('FUNCTION_NAME'):
        print("Using Application Default Credentials (GCP environment)")
        firebase_admin.initialize_app()
        return
    
    # Try JSON string from environment (for Vercel/Render/Heroku)
    firebase_creds_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
    if firebase_creds_json:
        print("Using Firebase credentials from environment variable")
        try:
            cred_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            return
        except json.JSONDecodeError as e:
            print(f"Error parsing FIREBASE_CREDENTIALS_JSON: {e}")
    
    # Try file path (for local development)
    firebase_creds_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
    if firebase_creds_path and os.path.exists(firebase_creds_path):
        print(f"Using Firebase credentials from file: {firebase_creds_path}")
        cred = credentials.Certificate(firebase_creds_path)
        firebase_admin.initialize_app(cred)
        return
    
    # Fallback
    print("Warning: No Firebase credentials found. Using default credentials.")
    firebase_admin.initialize_app()

# Initialize Firebase
initialize_firebase()

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'API is running'}), 200

# Predict endpoint for specific category
@app.route('/api/predict/<category>', methods=['POST'])
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
@app.route('/api/predict-all', methods=['POST'])
def predict_all():
    try:
        data = request.get_json() or {}
        years_ahead = data.get('years_ahead', 5)

        results = {}
        errors = {}

        for category in FIREBASE_COLLECTIONS.keys():
            try:
                predictor = Predictor(category)
                predictions = predictor.predict_future(years_ahead)
                results[category] = predictions
            except Exception as e:
                errors[category] = f"Prediction failed: {str(e)}"

        response = {'predictions': results}
        if errors:
            response['errors'] = errors
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# Train model for specific category
@app.route('/api/train/<category>', methods=['POST'])
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
@app.route('/api/model-info/<category>', methods=['GET'])
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
@app.route('/api/model-info-all', methods=['GET'])
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