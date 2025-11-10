"""
Data Preprocessing Module for House Price Prediction

This module handles data loading, cleaning, and preprocessing for the Ames Housing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for the Ames Housing dataset.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_features = []
        self.categorical_features = []
        
    def load_data(self, filepath):
        """
        Load the dataset from a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def identify_features(self, df):
        """
        Identify numerical and categorical features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        """
        self.numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove Id and SalePrice from features if present
        if 'Id' in self.numeric_features:
            self.numeric_features.remove('Id')
        if 'SalePrice' in self.numeric_features:
            self.numeric_features.remove('SalePrice')
            
        print(f"Numeric features: {len(self.numeric_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with handled missing values
        """
        df_clean = df.copy()
        
        # Features where NA means "None" or "No feature"
        none_features = [
            'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PoolQC', 'Fence', 'MiscFeature'
        ]
        
        for feature in none_features:
            if feature in df_clean.columns:
                df_clean[feature] = df_clean[feature].fillna('None')
        
        # Numeric features - fill with median
        for feature in self.numeric_features:
            if feature in df_clean.columns and df_clean[feature].isnull().sum() > 0:
                df_clean[feature] = df_clean[feature].fillna(df_clean[feature].median())
        
        # Categorical features - fill with mode
        for feature in self.categorical_features:
            if feature in df_clean.columns and df_clean[feature].isnull().sum() > 0:
                df_clean[feature] = df_clean[feature].fillna(df_clean[feature].mode()[0])
        
        print(f"Missing values handled. Remaining nulls: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def handle_outliers(self, df, target_col='SalePrice', method='IQR'):
        """
        Detect and handle outliers in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        method : str
            Method for outlier detection ('IQR' or 'zscore')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if target_col in df_clean.columns:
            # Remove extreme outliers in target variable
            Q1 = df_clean[target_col].quantile(0.25)
            Q3 = df_clean[target_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = df_clean[(df_clean[target_col] < lower_bound) | 
                               (df_clean[target_col] > upper_bound)]
            
            print(f"Detected {len(outliers)} outliers in {target_col}")
            
            # Remove extreme outliers
            df_clean = df_clean[(df_clean[target_col] >= lower_bound) & 
                               (df_clean[target_col] <= upper_bound)]
        
        return df_clean
    
    def encode_categorical(self, df, fit=True):
        """
        Encode categorical variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the encoders or use existing ones
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for feature in self.categorical_features:
            if feature in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                    self.label_encoders[feature] = le
                else:
                    if feature in self.label_encoders:
                        le = self.label_encoders[feature]
                        # Handle unseen categories
                        df_encoded[feature] = df_encoded[feature].apply(
                            lambda x: x if x in le.classes_ else le.classes_[0]
                        )
                        df_encoded[feature] = le.transform(df_encoded[feature].astype(str))
        
        return df_encoded
    
    def scale_features(self, df, features=None, fit=True):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        features : list
            List of features to scale (defaults to all numeric features)
        fit : bool
            Whether to fit the scaler or use existing one
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if features is None:
            features = [f for f in self.numeric_features if f in df_scaled.columns]
        
        if fit:
            df_scaled[features] = self.scaler.fit_transform(df_scaled[features])
        else:
            df_scaled[features] = self.scaler.transform(df_scaled[features])
        
        return df_scaled
    
    def preprocess(self, df, target_col='SalePrice', scale=False, handle_outliers_flag=True):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of the target column
        scale : bool
            Whether to scale features
        handle_outliers_flag : bool
            Whether to handle outliers
            
        Returns:
        --------
        tuple
            (X, y) - Features and target
        """
        # Identify feature types
        self.identify_features(df)
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Handle outliers if target is present
        if target_col in df_clean.columns and handle_outliers_flag:
            df_clean = self.handle_outliers(df_clean, target_col)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical(df_clean, fit=True)
        
        # Separate features and target
        if target_col in df_encoded.columns:
            y = df_encoded[target_col]
            X = df_encoded.drop([target_col], axis=1)
            
            # Remove Id if present
            if 'Id' in X.columns:
                X = X.drop(['Id'], axis=1)
        else:
            X = df_encoded.copy()
            if 'Id' in X.columns:
                X = X.drop(['Id'], axis=1)
            y = None
        
        # Scale features if requested
        if scale:
            X = self.scale_features(X, fit=True)
        
        print(f"Preprocessing complete. X shape: {X.shape}")
        if y is not None:
            print(f"Target shape: {y.shape}")
        
        return X, y


def get_feature_info(df):
    """
    Get comprehensive information about the dataset features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary containing feature information
    """
    info = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sort_values(ascending=False),
        'numeric_features': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
        'dtypes': df.dtypes
    }
    
    return info


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("=" * 50)
    print("\nThis module provides utilities for preprocessing the Ames Housing dataset.")
    print("\nKey features:")
    print("- Missing value handling")
    print("- Outlier detection and removal")
    print("- Categorical encoding")
    print("- Feature scaling")
    print("- Complete preprocessing pipeline")
