"""
Model Training Module for House Price Prediction

This module implements various machine learning models for price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class HousePricePredictor:
    """
    A comprehensive model training class for house price prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and validation sets.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        y : pd.Series or np.array
            Target variable
        test_size : float
            Proportion of data for validation
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_val, y_train, y_val)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        
        return X_train, X_val, y_train, y_val
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model."""
        print("\nTraining Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['LinearRegression'] = model
        return model
    
    def train_ridge(self, X_train, y_train, alpha=1.0):
        """Train Ridge Regression model."""
        print(f"\nTraining Ridge Regression (alpha={alpha})...")
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        self.models['Ridge'] = model
        return model
    
    def train_lasso(self, X_train, y_train, alpha=1.0):
        """Train Lasso Regression model."""
        print(f"\nTraining Lasso Regression (alpha={alpha})...")
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        self.models['Lasso'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=None, 
                           random_state=42):
        """Train Random Forest model."""
        print(f"\nTraining Random Forest (n_estimators={n_estimators})...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        return model
    
    def train_gradient_boosting(self, X_train, y_train, n_estimators=100, 
                                learning_rate=0.1, max_depth=3, random_state=42):
        """Train Gradient Boosting model."""
        print(f"\nTraining Gradient Boosting (n_estimators={n_estimators})...")
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        self.models['GradientBoosting'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, n_estimators=100, learning_rate=0.1,
                      max_depth=3, random_state=42):
        """Train XGBoost model."""
        print(f"\nTraining XGBoost (n_estimators={n_estimators})...")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['XGBoost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train, n_estimators=100, learning_rate=0.1,
                       max_depth=3, random_state=42):
        """Train LightGBM model."""
        print(f"\nTraining LightGBM (n_estimators={n_estimators})...")
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        self.models['LightGBM'] = model
        return model
    
    def evaluate_model(self, model, X_val, y_val, model_name="Model"):
        """
        Evaluate a model on validation data.
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_val : pd.DataFrame or np.array
            Validation features
        y_val : pd.Series or np.array
            Validation target
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        self.results[model_name] = metrics
        
        print(f"\n{model_name} Performance:")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE:  ${mae:,.2f}")
        print(f"  R²:   {r2:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """
        Train all models and evaluate them.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data
            
        Returns:
        --------
        pd.DataFrame
            Comparison of all models
        """
        print("=" * 60)
        print("Training All Models")
        print("=" * 60)
        
        # Train models
        self.train_linear_regression(X_train, y_train)
        self.train_ridge(X_train, y_train, alpha=10)
        self.train_lasso(X_train, y_train, alpha=100)
        self.train_random_forest(X_train, y_train, n_estimators=100, max_depth=20)
        self.train_gradient_boosting(X_train, y_train, n_estimators=100)
        self.train_xgboost(X_train, y_train, n_estimators=100)
        self.train_lightgbm(X_train, y_train, n_estimators=100)
        
        # Evaluate all models
        print("\n" + "=" * 60)
        print("Evaluating All Models")
        print("=" * 60)
        
        for name, model in self.models.items():
            self.evaluate_model(model, X_val, y_val, name)
        
        # Find best model
        results_df = pd.DataFrame(self.results).T
        self.best_model_name = results_df['RMSE'].idxmin()
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "=" * 60)
        print(f"Best Model: {self.best_model_name}")
        print("=" * 60)
        
        return results_df.sort_values('RMSE')
    
    def cross_validate(self, model, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        model : sklearn model
            Model to cross-validate
        X : pd.DataFrame or np.array
            Features
        y : pd.Series or np.array
            Target
        cv : int
            Number of folds
            
        Returns:
        --------
        dict
            Cross-validation scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        scores = cross_val_score(model, X, y, cv=cv, 
                                scoring='neg_root_mean_squared_error',
                                n_jobs=-1)
        
        cv_results = {
            'mean_score': -scores.mean(),
            'std_score': scores.std(),
            'scores': -scores
        }
        
        print(f"CV RMSE: ${-scores.mean():,.2f} (+/- ${scores.std() * 2:,.2f})")
        
        return cv_results
    
    def tune_hyperparameters(self, X_train, y_train, model_type='RandomForest'):
        """
        Tune hyperparameters using GridSearchCV.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        model_type : str
            Type of model to tune
            
        Returns:
        --------
        dict
            Best parameters
        """
        print(f"\nTuning hyperparameters for {model_type}...")
        
        if model_type == 'RandomForest':
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'XGBoost':
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        else:
            raise ValueError(f"Model type {model_type} not supported for tuning")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: ${-grid_search.best_score_:,.2f}")
        
        return grid_search.best_params_
    
    def save_model(self, model_name=None, filepath='models/best_model.pkl'):
        """
        Save a model to disk.
        
        Parameters:
        -----------
        model_name : str
            Name of model to save (uses best model if None)
        filepath : str
            Path to save the model
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
        
        if model is None:
            print(f"Model {model_name} not found!")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model, filepath)
        print(f"Model '{model_name}' saved to {filepath}")
    
    def load_model(self, filepath='models/best_model.pkl'):
        """
        Load a model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the model file
            
        Returns:
        --------
        sklearn model
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self, model_name=None, top_n=20):
        """
        Get feature importance from tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of model (uses best model if None)
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame
            Feature importance rankings
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature importances")
            return None
        
        importance_df = pd.DataFrame({
            'Feature': range(len(model.feature_importances_)),
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df


if __name__ == "__main__":
    # Example usage
    print("Model Training Module")
    print("=" * 50)
    print("\nThis module provides utilities for training ML models.")
    print("\nSupported models:")
    print("- Linear Regression")
    print("- Ridge Regression")
    print("- Lasso Regression")
    print("- Random Forest")
    print("- Gradient Boosting")
    print("- XGBoost")
    print("- LightGBM")
