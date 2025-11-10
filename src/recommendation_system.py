"""
House Recommendation System

This module provides personalized house recommendations based on customer preferences.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class HouseRecommendationSystem:
    """
    Recommendation system for suggesting houses based on customer preferences.
    """
    
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.knn_model = None
        self.feature_columns = []
    
    def load_data(self, data, target_col='SalePrice'):
        """
        Load house data for recommendations.
        
        Parameters:
        -----------
        data : pd.DataFrame
            House data with features and prices
        target_col : str
            Name of the price column
        """
        self.data = data.copy()
        self.target_col = target_col
        print(f"Loaded {len(self.data)} houses for recommendations")
    
    def filter_by_price(self, min_price=None, max_price=None):
        """
        Filter houses by price range.
        
        Parameters:
        -----------
        min_price : float
            Minimum price
        max_price : float
            Maximum price
            
        Returns:
        --------
        pd.DataFrame
            Filtered houses
        """
        filtered = self.data.copy()
        
        if min_price is not None:
            filtered = filtered[filtered[self.target_col] >= min_price]
        
        if max_price is not None:
            filtered = filtered[filtered[self.target_col] <= max_price]
        
        print(f"Price filter: {len(filtered)} houses match criteria")
        return filtered
    
    def filter_by_area(self, min_area=None, max_area=None, area_col='GrLivArea'):
        """
        Filter houses by living area.
        
        Parameters:
        -----------
        min_area : float
            Minimum area in square feet
        max_area : float
            Maximum area in square feet
        area_col : str
            Name of the area column
            
        Returns:
        --------
        pd.DataFrame
            Filtered houses
        """
        filtered = self.data.copy()
        
        if area_col not in filtered.columns:
            print(f"Warning: Column {area_col} not found")
            return filtered
        
        if min_area is not None:
            filtered = filtered[filtered[area_col] >= min_area]
        
        if max_area is not None:
            filtered = filtered[filtered[area_col] <= max_area]
        
        print(f"Area filter: {len(filtered)} houses match criteria")
        return filtered
    
    def filter_by_features(self, filters):
        """
        Filter houses by multiple feature criteria.
        
        Parameters:
        -----------
        filters : dict
            Dictionary of feature filters
            Example: {
                'BedroomAbvGr': {'min': 3, 'max': 5},
                'FullBath': {'min': 2},
                'GarageCars': {'min': 2}
            }
            
        Returns:
        --------
        pd.DataFrame
            Filtered houses
        """
        filtered = self.data.copy()
        
        for feature, criteria in filters.items():
            if feature not in filtered.columns:
                print(f"Warning: Feature {feature} not found")
                continue
            
            if 'min' in criteria:
                filtered = filtered[filtered[feature] >= criteria['min']]
            
            if 'max' in criteria:
                filtered = filtered[filtered[feature] <= criteria['max']]
            
            if 'equals' in criteria:
                filtered = filtered[filtered[feature] == criteria['equals']]
        
        print(f"Feature filters: {len(filtered)} houses match all criteria")
        return filtered
    
    def get_recommendations(self, preferences, n_recommendations=10):
        """
        Get house recommendations based on preferences.
        
        Parameters:
        -----------
        preferences : dict
            Customer preferences
            Example: {
                'max_price': 250000,
                'min_area': 1500,
                'max_area': 2500,
                'bedrooms': {'min': 3, 'max': 4},
                'bathrooms': {'min': 2},
                'garage': {'min': 2}
            }
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        pd.DataFrame
            Recommended houses
        """
        filtered = self.data.copy()
        
        # Apply price filter
        if 'min_price' in preferences or 'max_price' in preferences:
            filtered = self.filter_by_price(
                preferences.get('min_price'),
                preferences.get('max_price')
            )
            if filtered.empty:
                print("No houses match the price criteria")
                return pd.DataFrame()
        
        # Apply area filter
        if 'min_area' in preferences or 'max_area' in preferences:
            self.data = filtered
            filtered = self.filter_by_area(
                preferences.get('min_area'),
                preferences.get('max_area')
            )
            if filtered.empty:
                print("No houses match the area criteria")
                return pd.DataFrame()
        
        # Apply feature filters
        feature_filters = {}
        if 'bedrooms' in preferences:
            feature_filters['BedroomAbvGr'] = preferences['bedrooms']
        if 'bathrooms' in preferences:
            feature_filters['FullBath'] = preferences['bathrooms']
        if 'garage' in preferences:
            feature_filters['GarageCars'] = preferences['garage']
        
        if feature_filters:
            self.data = filtered
            filtered = self.filter_by_features(feature_filters)
            if filtered.empty:
                print("No houses match all feature criteria")
                return pd.DataFrame()
        
        # Sort by price (or other criteria) and return top N
        if self.target_col in filtered.columns:
            filtered = filtered.sort_values(self.target_col).head(n_recommendations)
        else:
            filtered = filtered.head(n_recommendations)
        
        print(f"\nReturning {len(filtered)} recommendations")
        return filtered
    
    def find_similar_houses(self, house_features, n_similar=5, feature_cols=None):
        """
        Find similar houses using k-NN algorithm.
        
        Parameters:
        -----------
        house_features : dict or pd.Series
            Features of the reference house
        n_similar : int
            Number of similar houses to find
        feature_cols : list
            List of features to use for similarity (optional)
            
        Returns:
        --------
        pd.DataFrame
            Similar houses
        """
        if feature_cols is None:
            # Use numeric columns
            feature_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_col in feature_cols:
                feature_cols.remove(self.target_col)
            if 'Id' in feature_cols:
                feature_cols.remove('Id')
        
        self.feature_columns = feature_cols
        
        # Prepare data
        X = self.data[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit k-NN model
        self.knn_model = NearestNeighbors(n_neighbors=n_similar + 1, metric='euclidean')
        self.knn_model.fit(X_scaled)
        
        # Prepare query features
        if isinstance(house_features, dict):
            query_features = pd.Series(house_features)[feature_cols].fillna(0).values.reshape(1, -1)
        else:
            query_features = house_features[feature_cols].fillna(0).values.reshape(1, -1)
        
        query_scaled = self.scaler.transform(query_features)
        
        # Find similar houses
        distances, indices = self.knn_model.kneighbors(query_scaled)
        
        # Return similar houses (excluding the first one if it's the same house)
        similar_houses = self.data.iloc[indices[0][1:]].copy()
        similar_houses['Similarity_Score'] = 1 / (1 + distances[0][1:])  # Inverse distance
        
        print(f"Found {len(similar_houses)} similar houses")
        return similar_houses
    
    def get_best_value_houses(self, n_houses=10, value_metric='price_per_sqft'):
        """
        Get best value houses based on price per square foot or other metrics.
        
        Parameters:
        -----------
        n_houses : int
            Number of houses to return
        value_metric : str
            Metric to use ('price_per_sqft', 'quality_price_ratio')
            
        Returns:
        --------
        pd.DataFrame
            Best value houses
        """
        df = self.data.copy()
        
        if value_metric == 'price_per_sqft':
            if 'GrLivArea' in df.columns and self.target_col in df.columns:
                df['PricePerSqFt'] = df[self.target_col] / df['GrLivArea']
                best_value = df.nsmallest(n_houses, 'PricePerSqFt')
            else:
                print("Required columns not found for price_per_sqft metric")
                return pd.DataFrame()
        
        elif value_metric == 'quality_price_ratio':
            if 'OverallQual' in df.columns and self.target_col in df.columns:
                df['QualityPriceRatio'] = df['OverallQual'] / (df[self.target_col] / 100000)
                best_value = df.nlargest(n_houses, 'QualityPriceRatio')
            else:
                print("Required columns not found for quality_price_ratio metric")
                return pd.DataFrame()
        
        else:
            print(f"Unknown value metric: {value_metric}")
            return pd.DataFrame()
        
        print(f"Found {len(best_value)} best value houses")
        return best_value
    
    def summarize_recommendations(self, recommendations):
        """
        Create a summary of recommended houses.
        
        Parameters:
        -----------
        recommendations : pd.DataFrame
            Recommended houses
            
        Returns:
        --------
        pd.DataFrame
            Summary of recommendations
        """
        if recommendations.empty:
            return pd.DataFrame()
        
        summary_cols = []
        
        # Include important columns if they exist
        important_cols = [
            'Id', self.target_col, 'GrLivArea', 'BedroomAbvGr', 
            'FullBath', 'HalfBath', 'GarageCars', 'YearBuilt', 
            'OverallQual', 'OverallCond', 'Neighborhood'
        ]
        
        for col in important_cols:
            if col in recommendations.columns:
                summary_cols.append(col)
        
        summary = recommendations[summary_cols].copy()
        
        # Add calculated fields if possible
        if 'GrLivArea' in summary.columns and self.target_col in summary.columns:
            summary['PricePerSqFt'] = summary[self.target_col] / summary['GrLivArea']
        
        return summary


def create_customer_profile(budget, desired_area, bedrooms, bathrooms, 
                           garage_size=None, min_quality=None):
    """
    Helper function to create a customer preference profile.
    
    Parameters:
    -----------
    budget : tuple or float
        (min_price, max_price) or max_price
    desired_area : tuple or float
        (min_area, max_area) or desired_area
    bedrooms : int or tuple
        Number of bedrooms or (min, max)
    bathrooms : int or tuple
        Number of bathrooms or (min, max)
    garage_size : int
        Minimum garage capacity
    min_quality : int
        Minimum overall quality rating
        
    Returns:
    --------
    dict
        Customer preference profile
    """
    profile = {}
    
    # Handle budget
    if isinstance(budget, tuple):
        profile['min_price'] = budget[0]
        profile['max_price'] = budget[1]
    else:
        profile['max_price'] = budget
    
    # Handle area
    if isinstance(desired_area, tuple):
        profile['min_area'] = desired_area[0]
        profile['max_area'] = desired_area[1]
    else:
        profile['min_area'] = desired_area * 0.8
        profile['max_area'] = desired_area * 1.2
    
    # Handle bedrooms
    if isinstance(bedrooms, tuple):
        profile['bedrooms'] = {'min': bedrooms[0], 'max': bedrooms[1]}
    else:
        profile['bedrooms'] = {'min': bedrooms, 'max': bedrooms}
    
    # Handle bathrooms
    if isinstance(bathrooms, tuple):
        profile['bathrooms'] = {'min': bathrooms[0], 'max': bathrooms[1]}
    else:
        profile['bathrooms'] = {'min': bathrooms}
    
    # Optional features
    if garage_size:
        profile['garage'] = {'min': garage_size}
    
    if min_quality:
        profile['quality'] = {'min': min_quality}
    
    return profile


if __name__ == "__main__":
    # Example usage
    print("House Recommendation System")
    print("=" * 50)
    print("\nThis module provides personalized house recommendations.")
    print("\nKey features:")
    print("- Price-based filtering")
    print("- Area-based filtering")
    print("- Multi-feature filtering")
    print("- Similar house finding (k-NN)")
    print("- Best value recommendations")
    print("- Customer preference profiles")
