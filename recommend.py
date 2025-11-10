#!/usr/bin/env python3
"""
House Recommendation System CLI

This script provides house recommendations based on customer preferences.
Usage: python recommend.py --max-price 200000 --min-area 1500 --bedrooms 3
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from recommendation_system import HouseRecommendationSystem


def main():
    parser = argparse.ArgumentParser(description='Get house recommendations')
    parser.add_argument('--data', type=str, default='data/train.csv',
                       help='Path to house data CSV file')
    parser.add_argument('--min-price', type=float, default=None,
                       help='Minimum price')
    parser.add_argument('--max-price', type=float, default=None,
                       help='Maximum price')
    parser.add_argument('--min-area', type=float, default=None,
                       help='Minimum living area (sq ft)')
    parser.add_argument('--max-area', type=float, default=None,
                       help='Maximum living area (sq ft)')
    parser.add_argument('--bedrooms', type=int, default=None,
                       help='Minimum number of bedrooms')
    parser.add_argument('--bathrooms', type=int, default=None,
                       help='Minimum number of bathrooms')
    parser.add_argument('--garage', type=int, default=None,
                       help='Minimum garage capacity (cars)')
    parser.add_argument('--num-recommendations', type=int, default=10,
                       help='Number of recommendations to return')
    parser.add_argument('--value-metric', type=str, default=None,
                       choices=['price_per_sqft', 'quality_price_ratio'],
                       help='Find best value houses using specified metric')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HOUSE RECOMMENDATION SYSTEM")
    print("=" * 70)
    
    # Load data
    print(f"\n1. Loading data from {args.data}...")
    try:
        df = pd.read_csv(args.data)
        print(f"   Loaded {len(df)} houses")
    except FileNotFoundError:
        print(f"   ERROR: {args.data} not found!")
        return
    
    # Initialize recommendation system
    print("\n2. Initializing recommendation system...")
    recommender = HouseRecommendationSystem()
    recommender.load_data(df, target_col='SalePrice')
    
    # Check if we're doing value-based recommendations
    if args.value_metric:
        print(f"\n3. Finding best value houses ({args.value_metric})...")
        recommendations = recommender.get_best_value_houses(
            n_houses=args.num_recommendations,
            value_metric=args.value_metric
        )
    else:
        # Build preferences
        print("\n3. Building customer preferences...")
        preferences = {}
        
        if args.min_price is not None:
            preferences['min_price'] = args.min_price
        if args.max_price is not None:
            preferences['max_price'] = args.max_price
        if args.min_area is not None:
            preferences['min_area'] = args.min_area
        if args.max_area is not None:
            preferences['max_area'] = args.max_area
        if args.bedrooms is not None:
            preferences['bedrooms'] = {'min': args.bedrooms}
        if args.bathrooms is not None:
            preferences['bathrooms'] = {'min': args.bathrooms}
        if args.garage is not None:
            preferences['garage'] = {'min': args.garage}
        
        # Display preferences
        print("\n   Customer Preferences:")
        if args.min_price or args.max_price:
            price_str = ""
            if args.min_price:
                price_str += f"${args.min_price:,.0f}"
            else:
                price_str += "Any"
            price_str += " - "
            if args.max_price:
                price_str += f"${args.max_price:,.0f}"
            else:
                price_str += "Any"
            print(f"   - Price Range: {price_str}")
        
        if args.min_area or args.max_area:
            area_str = ""
            if args.min_area:
                area_str += f"{args.min_area:,.0f}"
            else:
                area_str += "Any"
            area_str += " - "
            if args.max_area:
                area_str += f"{args.max_area:,.0f}"
            else:
                area_str += "Any"
            print(f"   - Living Area: {area_str} sq ft")
        
        if args.bedrooms:
            print(f"   - Bedrooms: {args.bedrooms}+")
        if args.bathrooms:
            print(f"   - Bathrooms: {args.bathrooms}+")
        if args.garage:
            print(f"   - Garage: {args.garage}+ cars")
        
        # Get recommendations
        print(f"\n4. Finding {args.num_recommendations} recommendations...")
        recommendations = recommender.get_recommendations(
            preferences,
            n_recommendations=args.num_recommendations
        )
    
    # Display results
    if recommendations.empty:
        print("\n" + "=" * 70)
        print("NO HOUSES MATCH YOUR CRITERIA")
        print("=" * 70)
        print("\nSuggestions:")
        print("- Try increasing your budget (--max-price)")
        print("- Relax area requirements")
        print("- Reduce minimum bedroom/bathroom requirements")
    else:
        print("\n" + "=" * 70)
        print(f"TOP {len(recommendations)} RECOMMENDATIONS")
        print("=" * 70)
        
        # Create summary
        summary = recommender.summarize_recommendations(recommendations)
        
        # Display as table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print("\n", summary.to_string(index=False))
        
        # Statistics
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Average Price: ${recommendations['SalePrice'].mean():,.2f}")
        print(f"Median Price: ${recommendations['SalePrice'].median():,.2f}")
        print(f"Price Range: ${recommendations['SalePrice'].min():,.2f} - ${recommendations['SalePrice'].max():,.2f}")
        
        if 'GrLivArea' in recommendations.columns:
            print(f"Average Area: {recommendations['GrLivArea'].mean():,.0f} sq ft")
        
        # Save to file
        output_file = 'recommendations.csv'
        summary.to_csv(output_file, index=False)
        print(f"\nRecommendations saved to: {output_file}")


if __name__ == "__main__":
    main()
