#!/usr/bin/env python3
"""
Apply the best models from hyperparameter tuning.
"""

import os
import shutil
import json

def main():
    backup_dir = 'models_candidates'
    models_dir = 'models'
    summary_path = os.path.join(backup_dir, 'tuning_summary.json')

    # Check if tuning was run
    if not os.path.exists(summary_path):
        print("❌ Error: No tuning results found.")
        print("     Run 'python hyperparameter_tuning.py' first.")
        return

    # Load summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print("=" * 60)
    print("Apply Best Models")
    print("=" * 60)

    print("\nBest configurations found: ")
    for category, data in summary.items():
        print(f"    - {category}: {data['best_config']} (MAE: {data['metrics']['val_mae']:.4f})")
    
    # Confirm
    response = input("\nReplace current models with the best ones? (y/n): ")

    if response.lower() not in ['y', 'yes']:
        print("Cancelled. No changes made.")
        return

    # Backup current models
    backup_current = 'models_backup_' + str(int(os.path.getmtime(models_dir)))
    print(f"\nBacking up current models to {backup_current}/")
    shutil.copytree(models_dir, backup_current)

    # Copy best models
    print("Copying best models to models/ directory...")
    for category in summary.keys():
        model_file = f"{category}_model.h5"
        scaler_file = f"{category}_scaler.pkl"

        shutil.copy(
            os.path.join(backup_dir, model_file),
            os.path.join(models_dir, model_file)
        )
        shutil.copy(
            os.path.join(backup_dir, scaler_file),
            os.path.join(models_dir, scaler_file)
        )
        print(f"✓ {category} model copied successfully!")
    
    # Update metadata
    metadata_path = os.path.join(models_dir, 'metadata.json')
    new_metadata = {}

    for category, data in summary.items():
        new_metadata[category] = {
            'last_trained': data['training_info'].get('last_trained', 'N/A'),
            'training_years': data['training_info']['years'],
            'num_years': data['training_info'].get('num_years', 0),
            'num_samples': data['training_info']['num_samples'],
            'num_features': data['training_info']['num_features'],
            'final_loss': data['metrics']['train_loss'],
            'final_mae': data['metrics']['train_mae'],
            'val_loss': data['metrics']['val_loss'],
            'val_mae': data['metrics']['val_mae'],
            'best_config': data['best_config'],
            'best_config_details': data['best_config_details'],
        }

    with open(metadata_path, 'w') as f:
        json.dump(new_metadata, f, indent=2)

    print("\n✓ Models successfully updated!")
    print(f"✓ Previous models backed up to: {backup_current}/")
    print("="*70)

if __name__ == '__main__':
    main()