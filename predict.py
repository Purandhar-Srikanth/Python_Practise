#!/usr/bin/env python3
"""
Make House Price Predictions

This script uses a trained model to make predictions on new data.
Usage: python predict.py --input data/test.csv --output predictions.csv
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse
import joblib

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(description='Predict house prices using trained model')
    parser.add_argument('--input', type=str, default='data/test.csv',
                       help='Path to input CSV file with house data')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Path to output CSV file for predictions')
    parser.add_argument('--model', type=str, default='models/best_model.pkl',
                       help='Path to trained model file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HOUSE PRICE PREDICTION")
    print("=" * 70)
    
    # Load input data
    print(f"\n1. Loading input data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"   Loaded {len(df)} houses")
    except FileNotFoundError:
        print(f"   ERROR: {args.input} not found!")
        return
    
    # Load trained model and processors
    print("\n2. Loading trained model and processors...")
    try:
        model = joblib.load(args.model)
        preprocessor = joblib.load('models/preprocessor.pkl')
        feature_engineer = joblib.load('models/feature_engineer.pkl')
        print("   ✓ Models loaded successfully")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print("   Please train a model first using train_model.py")
        return
    
    # Store the Id column if present
    ids = None
    if 'Id' in df.columns:
        ids = df['Id'].copy()
    
    # Preprocess data
    print("\n3. Preprocessing data...")
    # For test data, we don't have SalePrice, so handle it separately
    X_test, _ = preprocessor.preprocess(df.copy(), target_col='SalePrice' if 'SalePrice' in df.columns else None,
                                         scale=False, handle_outliers_flag=False)
    
    # Feature engineering
    print("\n4. Engineering features...")
    X_test_engineered = feature_engineer.create_all_features(X_test)
    
    # Ensure columns match training data
    # Get columns from training (we need to save these during training)
    # For now, we'll use the columns that exist
    print(f"   Feature count: {X_test_engineered.shape[1]}")
    
    # Make predictions
    print("\n5. Making predictions...")
    predictions = model.predict(X_test_engineered)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'Id': ids if ids is not None else range(len(predictions)),
        'SalePrice': predictions
    })
    
    # Save predictions
    print(f"\n6. Saving predictions to {args.output}...")
    output_df.to_csv(args.output, index=False)
    print("   ✓ Predictions saved successfully")
    
    # Display summary
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Number of predictions: {len(predictions)}")
    print(f"Mean predicted price: ${predictions.mean():,.2f}")
    print(f"Median predicted price: ${np.median(predictions):,.2f}")
    print(f"Min predicted price: ${predictions.min():,.2f}")
    print(f"Max predicted price: ${predictions.max():,.2f}")
    print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
