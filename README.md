# Filipino Emigrants ML Prediction API

A Flask-based RESTful API powered by TensorFlow LSTM (Long Short-Term Memory) neural networks for predicting future Filipino emigration trends across multiple demographic categories.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?logo=tensorflow&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-3.1.2-000000?logo=flask&logoColor=white) ![Firebase](https://img.shields.io/badge/Firebase-12.3.0-FFCA28?logo=firebase&logoColor=black)

## 🌐 Live API

**🔗 API Endpoint:** `http://localhost:5432` (local) or your deployed URL

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

## 🚀 Getting Started

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
├── render.yaml                  # Render deployment config
├── .env                         # Environment variables (not in git)
├── .gitignore
└── README.md
```

---

## 🔧 Usage

### Training Models

#### Train All Models

```bash
python train_all_models.py
```

This script will:
- Train models for all 7 categories
- Display progress and metrics
- Save models to `models/` directory
- Generate training summary

#### Train Specific Category via API

```bash
curl -X POST http://localhost:5432/api/train/age
```

### Making Predictions

#### Predict Specific Category

```bash
curl -X POST http://localhost:5432/api/predict/age \
  -H "Content-Type: application/json" \
  -d '{"years_ahead": 5}'
```

#### Predict All Categories

```bash
curl -X POST http://localhost:5432/api/predict-all \
  -H "Content-Type: application/json" \
  -d '{"years_ahead": 10}'
```

### Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

This script performs grid search to find optimal hyperparameters:
- LSTM layer sizes
- Dropout rates
- Learning rates
- Batch sizes

---

## 🌐 Deployment

### Deploy to Render

1. **Create a new Web Service** on [Render](https://render.com)

2. **Connect your repository**

3. **Configure the service**:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Region**: Choose closest to your users

4. **Add environment variables**:
   - `FIREBASE_CREDENTIALS_JSON`: Your Firebase service account JSON (as string)
   - `PYTHON_VERSION`: `3.12.0`

5. **Deploy** and note your API URL

6. **Update frontend** to use the deployed URL in `predictionService.ts`

### Deploy to Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add environment variables
railway variables set FIREBASE_CREDENTIALS_JSON='{"type":"service_account",...}'

# Deploy
railway up
```

### Deploy to Heroku

```bash
# Install Heroku CLI
# Login
heroku login

# Create app
heroku create your-app-name

# Add buildpack
heroku buildpacks:set heroku/python

# Set environment variables
heroku config:set FIREBASE_CREDENTIALS_JSON='{"type":"service_account",...}'

# Deploy
git push heroku main
```

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
        "https://your-frontend.vercel.app"
    ],
    supports_credentials=True
)
```

---

## 🧪 Testing

### Quick Test

```bash
python quick_test.py
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:5432/api/health

# Get model info
curl http://localhost:5432/api/model-info/age

# Make prediction
curl -X POST http://localhost:5432/api/predict/age \
  -H "Content-Type: application/json" \
  -d '{"years_ahead": 3}'
```

---

## 🆘 Troubleshooting

### Common Issues

#### Firebase Connection Error

**Problem:** "Firebase credentials not found"

**Solution:**
- Verify `.env` file exists in backend directory
- Check `FIREBASE_CREDENTIALS_PATH` or `FIREBASE_CREDENTIALS_JSON` is set
- Ensure Firebase credentials file exists at specified path
- For JSON string, ensure proper escaping

#### Model Not Found Error

**Problem:** "Model file not found for category: age"

**Solution:**
- Run `python train_all_models.py` to train models
- Check `models/` directory contains `.h5` and `.pkl` files
- Verify Firebase has historical data for the category

#### TensorFlow/CUDA Errors

**Problem:** GPU-related errors or warnings

**Solution:**
- TensorFlow will automatically use CPU if GPU unavailable
- For GPU support, install `tensorflow-gpu` and CUDA toolkit
- Warnings about GPU can be safely ignored for CPU-only deployment

#### Port Already in Use

**Problem:** "Address already in use: 5432"

**Solution:**
```bash
# Find process using port 5432
# Windows:
netstat -ano | findstr :5432

# macOS/Linux:
lsof -i :5432

# Kill the process or change port in app.py
```

#### Insufficient Data Error

**Problem:** "Not enough data to create sequences"

**Solution:**
- Ensure Firebase has at least 5 years of historical data
- Reduce `min_sequence_length` in `config.py`
- Check Firebase collection paths are correct

---

## 📊 Model Performance

### Typical Metrics

| Category | MAE | MSE | Training Time |
|----------|-----|-----|---------------|
| Age | 0.11 | 0.02 | ~2 min |
| Sex | 0.08 | 0.01 | ~1 min |
| Civil Status | 0.13 | 0.03 | ~2 min |
| Education | 0.10 | 0.02 | ~2 min |
| Occupation | 0.15 | 0.04 | ~3 min |
| Origin | 0.12 | 0.02 | ~2 min |
| Destination | 0.14 | 0.03 | ~3 min |

*Note: Metrics vary based on data quality and quantity*

---

## 🔄 Model Retraining

Models should be retrained when:

1. **New data available**: After adding new year's data to Firebase
2. **Performance degradation**: If predictions become less accurate
3. **Hyperparameter changes**: After tuning configuration
4. **Data distribution shift**: Significant changes in emigration patterns

### Automated Retraining (Recommended)

Set up a cron job or scheduled task:

```bash
# Linux/macOS crontab
0 0 1 * * cd /path/to/backend && /path/to/.venv/bin/python train_all_models.py

# Windows Task Scheduler
# Create task to run train_all_models.py monthly
```

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/ImprovedLSTM
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add improved LSTM architecture'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/ImprovedLSTM
   ```
5. **Open** a Pull Request

---

## 📄 License

This project is created for educational purposes as part of **ITD112 coursework**.

---

**🤖 Built with ❤️ for ITD112 Lab Assignment - ML Prediction API**
