#!/usr/bin/env python3
"""
Script to train all LSTM models for Filipino Emigrant Data prediction.
Run this script once initially to pre-train all category models.
"""

import firebase_admin
from firebase_admin import credentials
import os
from dotenv import load_dotenv
from ml.trainer import ModelTrainer
from ml.config import FIREBASE_COLLECTIONS

def main():
    print("=" * 60)
    print("Training all LSTM models for Filipino Emigrant Data prediction")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Initialize Firebase Admin
    cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
    if not cred_path or not os.path.exists(cred_path):
        print("Warning: Firebase credentials not found. Using application default credentials.")
        firebase_admin.initialize_app()
    else:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    
    # Train models for each category
    categories = list(FIREBASE_COLLECTIONS.keys())
    total = len(categories)

    results = {}

    for idx, category in enumerate(categories, 1):
        print(f"\n[{idx}/{total}] Training model for category: {category}")
        print("-" * 60)

        try:
            trainer = ModelTrainer(category)
            history = trainer.train(verbose=1)

            final_loss = history.history['loss'][-1]
            final_mae = history.history['mae'][-1]

            results[category] = {
                'status': 'success',
                'final_loss': final_loss,
                'final_mae': final_mae
            }

            print(f"✓ {category} model trained successfully!")

        except Exception as e:
            results[category] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"✗ {category} model training failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    failed_count = total - success_count

    print(f"Total categories: {total}")
    print(f"Successfully trained: {success_count} categories")
    print(f"Failed to train: {failed_count} categories")

    if failed_count > 0:
        print("\nFailed to train categories:")
        for cat, result in results.items():
            if result['status'] == 'failed':
                print(f"  - {cat}: {result['error']}")

    print("\nAll models trained and saved to 'models/' directory.")
    print("You can now use the Flask API server with 'python app.py' to predict and train models.")

if __name__ == '__main__':
    main()