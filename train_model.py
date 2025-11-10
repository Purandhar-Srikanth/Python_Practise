#!/usr/bin/env python3
"""
Train House Price Prediction Model

This script trains a machine learning model on the Ames Housing dataset.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import HousePricePredictor
import joblib


def main():
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    try:
        df = pd.read_csv('data/train.csv')
        print(f"   Loaded {len(df)} houses with {df.shape[1]} features")
    except FileNotFoundError:
        print("   ERROR: data/train.csv not found!")
        print("   Please download the dataset and place it in the data/ directory.")
        print("   See data/README.md for instructions.")
        return
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df, target_col='SalePrice', 
                                    scale=False, handle_outliers_flag=True)
    print(f"   Preprocessed data shape: {X.shape}")
    
    # Feature engineering
    print("\n3. Engineering features...")
    feature_engineer = FeatureEngineer()
    X_engineered = feature_engineer.create_all_features(X)
    print(f"   Final feature count: {X_engineered.shape[1]}")
    
    # Split data
    print("\n4. Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    
    # Train models
    print("\n5. Training models...")
    predictor = HousePricePredictor()
    results_df = predictor.train_all_models(X_train, y_train, X_val, y_val)
    
    # Display results
    print("\n6. Model Comparison:")
    print(results_df.to_string())
    
    print(f"\n7. Best Model: {predictor.best_model_name}")
    
    # Cross-validation
    print("\n8. Cross-validating best model...")
    cv_results = predictor.cross_validate(predictor.best_model, X_train, y_train, cv=5)
    print(f"   CV RMSE: ${cv_results['mean_score']:,.2f} (+/- ${cv_results['std_score'] * 2:,.2f})")
    
    # Save models
    print("\n9. Saving models and processors...")
    os.makedirs('models', exist_ok=True)
    
    predictor.save_model(filepath='models/best_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_engineer, 'models/feature_engineer.pkl')
    
    print("   ✓ Best model saved to models/best_model.pkl")
    print("   ✓ Preprocessor saved to models/preprocessor.pkl")
    print("   ✓ Feature engineer saved to models/feature_engineer.pkl")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest Model: {predictor.best_model_name}")
    print(f"Validation RMSE: ${results_df.loc[predictor.best_model_name, 'RMSE']:,.2f}")
    print(f"Validation R²: {results_df.loc[predictor.best_model_name, 'R2']:.4f}")


if __name__ == "__main__":
    main()
