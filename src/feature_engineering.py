"""
Feature Engineering Module for House Price Prediction

This module creates new features and selects important features for model training.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for creating and selecting features.
    """
    
    def __init__(self):
        self.feature_selector = None
        self.selected_features = None
    
    def create_age_features(self, df):
        """
        Create age-related features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with new age features
        """
        df_new = df.copy()
        
        if 'YearBuilt' in df_new.columns and 'YrSold' in df_new.columns:
            df_new['HouseAge'] = df_new['YrSold'] - df_new['YearBuilt']
        
        if 'YearRemodAdd' in df_new.columns and 'YrSold' in df_new.columns:
            df_new['YearsSinceRemodel'] = df_new['YrSold'] - df_new['YearRemodAdd']
        
        if 'GarageYrBlt' in df_new.columns and 'YrSold' in df_new.columns:
            df_new['GarageAge'] = df_new['YrSold'] - df_new['GarageYrBlt']
            df_new['GarageAge'] = df_new['GarageAge'].fillna(0)
        
        return df_new
    
    def create_area_features(self, df):
        """
        Create total area and area-related features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with new area features
        """
        df_new = df.copy()
        
        # Total square footage
        area_cols = []
        if 'TotalBsmtSF' in df_new.columns:
            area_cols.append('TotalBsmtSF')
        if '1stFlrSF' in df_new.columns:
            area_cols.append('1stFlrSF')
        if '2ndFlrSF' in df_new.columns:
            area_cols.append('2ndFlrSF')
        
        if area_cols:
            df_new['TotalSF'] = df_new[area_cols].sum(axis=1)
        
        # Total bathrooms
        bath_cols = []
        if 'FullBath' in df_new.columns:
            bath_cols.append(('FullBath', 1))
        if 'HalfBath' in df_new.columns:
            bath_cols.append(('HalfBath', 0.5))
        if 'BsmtFullBath' in df_new.columns:
            bath_cols.append(('BsmtFullBath', 1))
        if 'BsmtHalfBath' in df_new.columns:
            bath_cols.append(('BsmtHalfBath', 0.5))
        
        if bath_cols:
            df_new['TotalBaths'] = sum(df_new[col] * weight for col, weight in bath_cols)
        
        # Total porch area
        porch_cols = []
        for col in ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']:
            if col in df_new.columns:
                porch_cols.append(col)
        
        if porch_cols:
            df_new['TotalPorchSF'] = df_new[porch_cols].sum(axis=1)
        
        return df_new
    
    def create_quality_features(self, df):
        """
        Create quality and condition interaction features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with new quality features
        """
        df_new = df.copy()
        
        if 'OverallQual' in df_new.columns and 'OverallCond' in df_new.columns:
            df_new['OverallScore'] = df_new['OverallQual'] * df_new['OverallCond']
        
        if 'OverallQual' in df_new.columns and 'GrLivArea' in df_new.columns:
            df_new['QualityArea'] = df_new['OverallQual'] * df_new['GrLivArea']
        
        if 'ExterQual' in df_new.columns and 'ExterCond' in df_new.columns:
            df_new['ExterScore'] = df_new['ExterQual'] * df_new['ExterCond']
        
        return df_new
    
    def create_boolean_features(self, df):
        """
        Create boolean features for certain conditions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with new boolean features
        """
        df_new = df.copy()
        
        if '2ndFlrSF' in df_new.columns:
            df_new['Has2ndFloor'] = (df_new['2ndFlrSF'] > 0).astype(int)
        
        if 'GarageArea' in df_new.columns:
            df_new['HasGarage'] = (df_new['GarageArea'] > 0).astype(int)
        
        if 'TotalBsmtSF' in df_new.columns:
            df_new['HasBasement'] = (df_new['TotalBsmtSF'] > 0).astype(int)
        
        if 'Fireplaces' in df_new.columns:
            df_new['HasFireplace'] = (df_new['Fireplaces'] > 0).astype(int)
        
        if 'PoolArea' in df_new.columns:
            df_new['HasPool'] = (df_new['PoolArea'] > 0).astype(int)
        
        return df_new
    
    def create_all_features(self, df):
        """
        Create all engineered features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all new features
        """
        print("Creating engineered features...")
        
        df_new = df.copy()
        df_new = self.create_age_features(df_new)
        df_new = self.create_area_features(df_new)
        df_new = self.create_quality_features(df_new)
        df_new = self.create_boolean_features(df_new)
        
        new_features = [col for col in df_new.columns if col not in df.columns]
        print(f"Created {len(new_features)} new features: {new_features}")
        
        return df_new
    
    def select_features(self, X, y, k=50, method='f_regression'):
        """
        Select top k features based on statistical tests.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        k : int
            Number of features to select
        method : str
            Feature selection method ('f_regression' or 'mutual_info')
            
        Returns:
        --------
        pd.DataFrame
            Selected features
        """
        if method == 'f_regression':
            score_func = f_regression
        elif method == 'mutual_info':
            score_func = mutual_info_regression
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.feature_selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        mask = self.feature_selector.get_support()
        self.selected_features = X.columns[mask].tolist()
        
        print(f"Selected {len(self.selected_features)} features using {method}")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def get_feature_scores(self, X, y, method='f_regression'):
        """
        Get importance scores for all features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        method : str
            Scoring method
            
        Returns:
        --------
        pd.DataFrame
            Feature scores sorted by importance
        """
        if method == 'f_regression':
            scores = f_regression(X, y)[0]
        elif method == 'mutual_info':
            scores = mutual_info_regression(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores
        }).sort_values('Score', ascending=False)
        
        return feature_scores


def create_polynomial_features(df, features, degree=2):
    """
    Create polynomial features for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : list
        List of features to create polynomials for
    degree : int
        Polynomial degree
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with polynomial features added
    """
    df_new = df.copy()
    
    for feature in features:
        if feature in df_new.columns:
            for d in range(2, degree + 1):
                df_new[f'{feature}_pow{d}'] = df_new[feature] ** d
    
    return df_new


def create_interaction_features(df, feature_pairs):
    """
    Create interaction features between feature pairs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_pairs : list of tuples
        Pairs of features to create interactions for
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with interaction features added
    """
    df_new = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df_new.columns and feat2 in df_new.columns:
            df_new[f'{feat1}_x_{feat2}'] = df_new[feat1] * df_new[feat2]
    
    return df_new


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("=" * 50)
    print("\nThis module provides utilities for feature engineering.")
    print("\nKey features:")
    print("- Age-related features")
    print("- Area aggregation features")
    print("- Quality interaction features")
    print("- Boolean indicator features")
    print("- Feature selection")
    print("- Polynomial features")
    print("- Interaction features")
