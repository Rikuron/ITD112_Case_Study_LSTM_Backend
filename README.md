---
title: Filipino Emigrants ML API
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Filipino Emigrants ML Prediction API

A Flask-based RESTful API powered by TensorFlow LSTM (Long Short-Term Memory) neural networks for predicting future Filipino emigration trends across multiple demographic categories.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?logo=tensorflow&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-3.1.2-000000?logo=flask&logoColor=white) ![Firebase](https://img.shields.io/badge/Firebase-12.3.0-FFCA28?logo=firebase&logoColor=black) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface&logoColor=black)

## 🌐 Live API

**🔗 Public Endpoint:** `https://sh00py-itd-112-filipino-emigrants-lstm.hf.space`

> [!NOTE]
> This is a public demo endpoint.

---

## ✨ Features

### 🤖 **Machine Learning Capabilities**

- **LSTM Neural Networks**: Deep learning models for time-series forecasting
- **Multi-category Predictions**: Separate models for 7 demographic categories
- **Automated Training**: Train models on historical Firebase data
- **Hyperparameter Tuning**: Optimize model performance with grid search
- **Model Persistence**: Save and load trained models efficiently
- **Real-time Predictions**: Fast inference for future trend forecasting

### 📊 **Supported Categories**

- **Age Groups**: 14 age brackets prediction
- **Sex Distribution**: Male/Female emigration trends
- **Civil Status**: Marital status distribution forecasting
- **Education Levels**: Educational attainment predictions
- **Occupation Categories**: Professional distribution trends
- **Origin Regions**: Regional emigration patterns
- **Destination Countries**: Major destination forecasting

### 🔥 **Firebase Integration**

- **Real-time Data Fetching**: Retrieve historical data from Firestore
- **Flexible Authentication**: Support for multiple credential methods
- **Secure Access**: Firebase Admin SDK integration

### 🚀 **RESTful API**

- **Health Monitoring**: API status endpoint
- **Prediction Endpoints**: Category-specific and bulk predictions
- **Training Endpoints**: On-demand model training
- **Model Information**: Access model metadata and performance metrics
- **CORS Support**: Cross-origin requests for frontend integration

---

## 🛠️ Tech Stack

### Core Framework

- **Flask 3.1.2** - Lightweight WSGI web application framework
- **Flask-CORS 6.0.1** - Cross-Origin Resource Sharing support
- **Gunicorn 23.0.0** - Production WSGI HTTP server

### Machine Learning

- **TensorFlow 2.20.0** - Deep learning framework
- **LSTM Architecture** - Recurrent neural networks for time-series
- **NumPy 2.3.4** - Numerical computing library
- **Pandas 2.3.3** - Data manipulation and analysis
- **Scikit-learn 1.7.2** - Data preprocessing and scaling

### Backend Services

- **Firebase Admin 6.5.0** - Firebase Admin SDK for Python
- **Python-dotenv 1.0.1** - Environment variable management

---

## 📋 API Endpoints

### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

### Predict Specific Category

```http
POST /api/predict/<category>
```

**Categories:** `age`, `sex`, `civil_status`, `education`, `occupation`, `origin`, `destination`

**Request Body:**
```json
{
  "years_ahead": 5
}
```

**Response:**
```json
{
  "category": "age",
  "predictions": [
    {
      "year": 2024,
      "14 - Below": 120.5,
      "15-19": 450.2,
      ...
    },
    ...
  ]
}
```

### Predict All Categories

```http
POST /api/predict-all
```

**Request Body:**
```json
{
  "years_ahead": 5
}
```

**Response:**
```json
{
  "predictions": {
    "age": [...],
    "sex": [...],
    "civil_status": [...],
    ...
  },
  "errors": {}
}
```

### Train Model

```http
POST /api/train/<category>
```

**Response:**
```json
{
  "message": "Model trained successfully for category: age",
  "final_loss": 0.0234,
  "final_mae": 0.1123
}
```

### Get Model Information

```http
GET /api/model-info/<category>
```

**Response:**
```json
{
  "category": "age",
  "trained_at": "2024-11-27T15:30:00",
  "num_features": 14,
  "sequence_length": 5,
  "final_loss": 0.0234,
  "final_mae": 0.1123,
  "epochs_trained": 45
}
```

### Get All Models Information

```http
GET /api/model-info-all
```

**Response:**
```json
{
  "models": {
    "age": {...},
    "sex": {...},
    ...
  }
}
```

---

## 🧠 Model Architecture

### LSTM Configuration

```python
LSTM_CONFIG = {
    'units_layer1': 64,      # First LSTM layer units
    'units_layer2': 32,      # Second LSTM layer units
    'dropout_rate': 0.2,     # Dropout for regularization
    'epochs': 100,           # Maximum training epochs
    'batch_size': 8,         # Training batch size
    'validation_split': 0.2, # Validation data split
    'patience': 15,          # Early stopping patience
    'learning_rate': 0.001   # Adam optimizer learning rate
}
```

### Network Architecture

```
Input Layer (sequence_length, num_features)
    ↓
LSTM Layer 1 (64 units, tanh activation)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (32 units, tanh activation)
    ↓
Dropout (0.2)
    ↓
Dense Output Layer (num_features, linear activation)
```

### Training Process

1. **Data Collection**: Fetch historical data from Firebase Firestore
2. **Preprocessing**: 
   - Normalize data using MinMaxScaler
   - Create time-series sequences (default: 5 years lookback)
3. **Model Training**:
   - Adam optimizer with learning rate 0.001
   - Mean Squared Error (MSE) loss function
   - Mean Absolute Error (MAE) metric
   - Early stopping with patience of 15 epochs
4. **Validation**: 20% of data used for validation
5. **Model Saving**: Save model (.h5) and scaler (.pkl) to disk
6. **Metadata Storage**: Store training metrics in JSON

---

## 🚀 Getting Started (Local Development)

### Prerequisites

- **Python 3.8+** (recommended: 3.12)
- **pip** package manager
- **Firebase Project** with Firestore database
- **Firebase Admin SDK credentials**

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd lab1/backend
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the backend directory:

   ```env
   # Option 1: Use file path
   FIREBASE_CREDENTIALS_PATH=path/to/your-firebase-credentials.json
   
   # Option 2: Use JSON string (for deployment)
   FIREBASE_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project",...}
   ```

5. **Train models** (first-time setup)

   ```bash
   python train_all_models.py
   ```

   This will:
   - Connect to Firebase Firestore
   - Fetch historical data for all categories
   - Train LSTM models for each category
   - Save models to `models/` directory
   - Generate metadata.json with training metrics

6. **Start the Flask server**

   ```bash
   python app.py
   ```

   The API will be available at `http://localhost:5432`

---

## 📁 Project Structure

```
backend/
├── ml/                          # Machine learning modules
│   ├── __init__.py
│   ├── config.py                # Configuration and hyperparameters
│   ├── data_preprocessor.py     # Data fetching and preprocessing
│   ├── lstm_model.py            # LSTM model architecture
│   ├── predictor.py             # Prediction logic
│   └── trainer.py               # Model training logic
│
├── models/                      # Saved trained models
│   ├── age_model.h5             # Age category model
│   ├── age_scaler.pkl           # Age category scaler
│   ├── sex_model.h5
│   ├── sex_scaler.pkl
│   ├── ...                      # Other category models
│   └── metadata.json            # Training metadata
│
├── app.py                       # Flask API server
├── train_all_models.py          # Script to train all models
├── hyperparameter_tuning.py     # Hyperparameter optimization
├── quick_test.py                # Quick testing script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration for Hugging Face
├── .env                         # Environment variables (not in git)
├── .gitignore
└── README.md
```

---

## 🌐 Deployment to Hugging Face Spaces

This API is configured for deployment on [Hugging Face Spaces](https://huggingface.co/spaces) using the Docker SDK.

### Steps to Deploy

1.  **Create a New Space**:
    -   Go to Hugging Face Spaces and create a new Space.
    -   Select **Docker** as the SDK.

2.  **Push Code**:
    -   Push this `backend` directory to your Space's repository.

3.  **Configure Secrets**:
    -   In your Space's **Settings**, go to the **Variables and secrets** section.
    -   Add a new Secret:
        -   **Name**: `FIREBASE_CREDENTIALS_JSON`
        -   **Value**: Paste the content of your Firebase service account JSON file.

4.  **Build and Run**:
    -   Hugging Face will automatically build the Docker image and start the application.
    -   The `Dockerfile` is configured to expose port `7860` and run the Flask app with Gunicorn.

---

## 🔒 Security Considerations

### Development

- ✅ Keep `.env` file in `.gitignore`
- ✅ Use environment variables for credentials
- ✅ Test with local Firebase emulator when possible

### Production

- ⚠️ **Never commit** Firebase credentials to version control
- ⚠️ Use **environment variables** for all sensitive data
- ⚠️ Enable **Firebase App Check** for additional security
- ⚠️ Implement **rate limiting** for API endpoints
- ⚠️ Use **HTTPS** for all API communications
- ⚠️ Restrict **CORS** to trusted domains only
- ⚠️ Monitor **API usage** and set quotas

### CORS Configuration

Update `app.py` for production:

```python
CORS(
    app, 
    origins=[
        "https://your-frontend-domain.com",
        "https://your-frontend.vercel.app",
        r"https://.*\.hf\.space"
    ],
    supports_credentials=True
)
```

---

## 📄 License

This project is created for educational purposes as part of **ITD112 coursework**.

---

**🤖 Built with ❤️ for ITD112 Lab Assignment - ML Prediction API**
