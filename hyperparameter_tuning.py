#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for LSTM Models
Trains multiple models with different configurations and selects the best one.
"""

import firebase_admin
from firebase_admin import credentials
import os
from dotenv import load_dotenv
from ml.data_preprocessor import DataProcessor
from ml.lstm_model import LSTMModel
from ml.config import FIREBASE_COLLECTIONS, MODELS_DIR
import json
from datetime import datetime
import shutil

CONFIG_VARIANTS = {
    # === BASELINE (Current Winner) ===
    'baseline': {
        'name': 'Baseline (64/32)',
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.001,
        'sequence_length': 15
    },
    
    # === SMALLER MODELS (Test Simplicity) ===
    'tiny': {
        'name': 'Tiny (32/16)',
        'units_layer1': 32,
        'units_layer2': 16,
        'dropout_rate': 0.15,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.001,
        'sequence_length': 15
    },
    
    'micro': {
        'name': 'Micro (48/24)',
        'units_layer1': 48,
        'units_layer2': 24,
        'dropout_rate': 0.18,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.001,
        'sequence_length': 15
    },
    
    'compact': {
        'name': 'Compact (56/28)',
        'units_layer1': 56,
        'units_layer2': 28,
        'dropout_rate': 0.19,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.001,
        'sequence_length': 15
    },
    
    # === SLIGHTLY LARGER (Just Above Baseline) ===
    'baseline_plus': {
        'name': 'Baseline+ (80/40)',
        'units_layer1': 80,
        'units_layer2': 40,
        'dropout_rate': 0.22,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.001,
        'sequence_length': 15
    },
    
    # === SEQUENCE LENGTH VARIATIONS ===
    'baseline_short_seq': {
        'name': 'Baseline (Seq=10)',
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.001,
        'sequence_length': 10
    },
    
    'baseline_long_seq': {
        'name': 'Baseline (Seq=20)',
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.001,
        'sequence_length': 20
    },
    
    # === DROPOUT VARIATIONS ===
    'baseline_less_dropout': {
        'name': 'Baseline (Dropout=0.15)',
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.15,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.001,
        'sequence_length': 15
    },
    
    'baseline_more_dropout': {
        'name': 'Baseline (Dropout=0.25)',
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.25,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.001,
        'sequence_length': 15
    },
    
    # === LEARNING RATE VARIATIONS ===
    'baseline_slower_lr': {
        'name': 'Baseline (LR=0.0005)',
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.2,
        'epochs': 120,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 20,
        'learning_rate': 0.0005,
        'sequence_length': 15
    },
    
    'baseline_faster_lr': {
        'name': 'Baseline (LR=0.002)',
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 15,
        'learning_rate': 0.002,
        'sequence_length': 15
    },
    
    # === TRAINING DURATION ===
    'baseline_longer_training': {
        'name': 'Baseline (Epochs=150)',
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.2,
        'epochs': 150,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 25,
        'learning_rate': 0.001,
        'sequence_length': 15
    },
    
    # === BEST OF SMALL (Optimized Tiny) ===
    'tiny_optimized': {
        'name': 'Tiny Optimized (32/16, Tuned)',
        'units_layer1': 32,
        'units_layer2': 16,
        'dropout_rate': 0.12,
        'epochs': 120,
        'batch_size': 8,
        'validation_split': 0.15,
        'patience': 20,
        'learning_rate': 0.0008,
        'sequence_length': 12
    }
}

def train_with_config(category, config_name, config):
    """Train a model with specific configuration."""
    print(f"\n  Testing config: {config['name']}")
    print(f"    Units: {config['units_layer1']}/{config['units_layer2']}")
    print(f"    Dropout: {config['dropout_rate']}, LR: {config['learning_rate']}")
    print(f"    Sequence Length: {config['sequence_length']}")

    try:
        # Prepare data with custom sequence length
        data_processor = DataProcessor(category)
        df = data_processor.fetch_data_from_firebase()
        years, data_normalized = data_processor.preprocess_data(df)

        # Create sequences with config-specific length
        X, y = data_processor.create_sequences(data_normalized, config['sequence_length'])

        if X.shape[0] < 2:
            print(f"    ⚠️  Skipped: Not enough samples ({X.shape[0]})")
            return None
        
        print(f"    Training Shape: X = {X.shape}, y = {y.shape}")

        # Build and train model
        input_shape = (X.shape[1], X.shape[2])
        output_shape = y.shape[1]

        model = LSTMModel(input_shape, output_shape)

        # Override model's build to use current config
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Rebuild model with current config
        new_model = keras.Sequential([
            layers.LSTM(
                config['units_layer1'],
                activation='tanh',
                return_sequences=True,
                input_shape=input_shape
            ),
            layers.Dropout(config['dropout_rate']),

            layers.LSTM(
                config['units_layer2'],
                activation='tanh',
                return_sequences=False,
            ),
            layers.Dropout(config['dropout_rate']),

            layers.Dense(output_shape, activation='linear')
        ])

        optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
        new_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )

        model.model = new_model

        # Determine if we should use validation
        num_samples = X.shape[0]
        use_validation = num_samples >= 10 and config['validation_split'] > 0

        callbacks = []
        if use_validation:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config['patience'],
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stopping)
            validation_split = config['validation_split']
        else:
            validation_split = 0.0
            
        # Train
        history = new_model.fit(
            X, y,
            epochs=config['epochs'],
            batch_size=min(config['batch_size'], num_samples),
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        # Get metrics
        final_loss = float(history.history['loss'][-1])
        final_mae = float(history.history['mae'][-1])

        # Check if validation was used
        has_val = 'val_loss' in history.history
        if has_val:
            val_loss = float(history.history['val_loss'][-1])
            val_mae = float(history.history['val_mae'][-1])
        else:
            val_loss = final_loss
            val_mae = final_mae

        result = {
            'config_name': config_name,
            'config': config,
            'model': model,
            'data_processor': data_processor,
            'metrics': {
                'train_loss': final_loss,
                'train_mae': final_mae,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'has_validation': has_val
            },
            'training_info': {
                'num_samples': int(X.shape[0]),
                'num_features': int(X.shape[2]),
                'years': f"{years[0]} - {years[-1]}",
            }
        }

        print(f"    ✓ Train Loss: {final_loss:.4f}, MAE: {final_mae:.4f}")
        if has_val:
            print(f"    ✓ Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

        return result

    except Exception as e:
        print(f"    ✗ Training failed: {str(e)}")
        return None

def select_best_model(results):
    """Select the best model based on metrics."""
    if not results:
        return None

    # Sort by validation MAE (or training MAE if no validation)
    results.sort(key = lambda x: x['metrics']['val_mae'] if x['metrics']['has_validation'] else x['metrics']['train_mae'])
    return results[0]

def main():
    print("=" * 60)
    print("LSTM Hyperparameter Tuning")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Initialize firebase
    cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
    if cred_path and os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app()

    # Create backup directory
    backup_dir = 'models_candidates'
    os.makedirs(backup_dir, exist_ok=True)

    # Results storage
    all_results = {}

    # Test each category
    categories = list(FIREBASE_COLLECTIONS.keys())
    total_categories = len(categories)
    total_configs = len(CONFIG_VARIANTS)

    print(f"\nTesting {total_configs} configurations for {total_categories} categories")
    print(f"Total training runs: {total_categories * total_configs}")
    print(f"\nThis will take a while...")

    for cat_idx, category in enumerate(categories, 1):
        print(f"\n{'=' * 60}")
        print(f"[{cat_idx}/{total_categories}] Testing category: {category.upper()}")
        print("-" * 60)

        category_results = []

        # Test each configuration
        for config_idx, (config_name, config) in enumerate(CONFIG_VARIANTS.items(), 1):
            print(f"\n  [{config_idx}/{total_configs}] ", end='')
            result = train_with_config(category, config_name, config)

            if result:
                category_results.append(result)

        # Select best model for this category
        if category_results:
            best = select_best_model(category_results)
            all_results[category] = {
                'best': best,
                'all_results': category_results
            }

            print(f"\n  🏆 BEST CONFIG for {category.upper()}: {best['config']['name']}")
            print(f"    VAL MAE: {best['metrics']['val_mae']:.4f}")
            print(f"    TRAIN MAE: {best['metrics']['train_mae']:.4f}")

            # Save best model temporarily
            temp_model_path = os.path.join(backup_dir, f"{category}_model.h5")
            temp_scaler_path = os.path.join(backup_dir, f"{category}_scaler.pkl")
            best['model'].save(temp_model_path)
            best['data_processor'].save_scaler(temp_scaler_path)

    # Summary Report
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 60)

    summary_data = {}

    for category, data in all_results.items():
        best = data['best']
        all_configs = data['all_results']

        print(f"\n{category.upper()}: ")
        print(f"    Best Config: {best['config']['name']}")
        print(f"    Best VAL MAE: {best['metrics']['val_mae']:.4f}")
        print(f"    Configurations tested: {len(all_configs)}")

        # Compare with baseline
        baseline = next((r for r in all_configs if r['config_name'] == 'baseline'), None)
        if baseline:
            improvement = ((baseline['metrics']['val_mae'] - best['metrics']['val_mae'])
                            / baseline['metrics']['val_mae'] * 100)
            print(f"    Improvement over baseline: {improvement:.1f}%")

        # Show all tested configs
        print(f"\n  All results (sorted by VAL MAE): ")
        for idx, result in enumerate(sorted(all_configs, key = lambda x: x['metrics']['val_mae']), 1):
            print(f"    {idx}. {result['config']['name']:<30} "
                    f"MAE: {result['metrics']['val_mae']:.4f}")

        summary_data[category] = {
            'best_config': best['config_name'],
            'best_config_details': best['config'],
            'metrics': best['metrics'],
            'training_info': best['training_info']
        }
    
    # Save summary
    summary_path = os.path.join(backup_dir, 'tuning_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Results saved to: ")
    print(f"    - Models: {backup_dir}/")
    print(f"    - SUmmary: {summary_path}")
    print('=' * 60)

    # Ask user to confirm before replacing current models
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Review the results above")
    print("2. Check tuning_summary.json for detailed metrics")
    print("3. If satisfied, run: python apply_best_models.py")
    print("   This will replace your current models with the best ones")
    print("\n ⚠️ Current models in 'models/' folder are NOT modified yet")
    print("="*70)
    
if __name__ == '__main__':
    main()