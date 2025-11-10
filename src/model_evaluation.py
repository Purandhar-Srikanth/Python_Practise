"""
Model Evaluation Module for House Price Prediction

This module provides utilities for evaluating and visualizing model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Median Absolute Error
    median_ae = np.median(np.abs(y_true - y_pred))
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Median_AE': median_ae
    }
    
    return metrics


def print_metrics(metrics, title="Model Evaluation Metrics"):
    """
    Print evaluation metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    title : str
        Title for the metrics output
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for name, value in metrics.items():
        if name in ['RMSE', 'MAE', 'Median_AE']:
            print(f"{name:15s}: ${value:,.2f}")
        elif name == 'MAPE':
            print(f"{name:15s}: {value:.2f}%")
        else:
            print(f"{name:15s}: {value:.4f}")
    print("=" * 60)


def plot_predictions(y_true, y_pred, title="Predicted vs Actual Prices", 
                     save_path=None):
    """
    Plot predicted vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str
        Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction')
    
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    """
    Plot residuals to check for patterns.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str
        Path to save the plot (optional)
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual scatter plot
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[0].set_ylabel('Residuals ($)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals ($)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_importance_df, title="Feature Importance", 
                           top_n=20, save_path=None):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    title : str
        Plot title
    top_n : int
        Number of top features to display
    save_path : str
        Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    top_features = feature_importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results_df, metric='RMSE', title="Model Comparison", 
                         save_path=None):
    """
    Plot comparison of different models.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model results
    metric : str
        Metric to compare
    title : str
        Plot title
    save_path : str
        Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    results_sorted = results_df.sort_values(metric)
    
    colors = ['green' if x == results_sorted[metric].min() else 'skyblue' 
              for x in results_sorted[metric]]
    
    plt.bar(range(len(results_sorted)), results_sorted[metric], color=colors, 
            edgecolor='black')
    plt.xticks(range(len(results_sorted)), results_sorted.index, rotation=45, ha='right')
    plt.ylabel(metric, fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title(f'{title} - {metric}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(results_sorted[metric]):
        plt.text(i, v, f'${v:,.0f}' if metric in ['RMSE', 'MAE'] else f'{v:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_learning_curve(train_scores, val_scores, title="Learning Curve", 
                       save_path=None):
    """
    Plot learning curves to diagnose bias/variance.
    
    Parameters:
    -----------
    train_scores : list
        Training scores over iterations
    val_scores : list
        Validation scores over iterations
    title : str
        Plot title
    save_path : str
        Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_scores, label='Training Score', linewidth=2)
    plt.plot(val_scores, label='Validation Score', linewidth=2)
    
    plt.xlabel('Iteration / Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def create_evaluation_report(y_true, y_pred, model_name="Model", save_dir=None):
    """
    Create a comprehensive evaluation report with plots and metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
    save_dir : str
        Directory to save plots (optional)
    """
    print(f"\n{'=' * 70}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'=' * 70}\n")
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, f"{model_name} - Evaluation Metrics")
    
    # Create plots
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        pred_path = f"{save_dir}/{model_name}_predictions.png"
        resid_path = f"{save_dir}/{model_name}_residuals.png"
    else:
        pred_path = None
        resid_path = None
    
    plot_predictions(y_true, y_pred, 
                    title=f"{model_name} - Predicted vs Actual",
                    save_path=pred_path)
    
    plot_residuals(y_true, y_pred,
                  title=f"{model_name} - Residual Analysis", 
                  save_path=resid_path)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("=" * 50)
    print("\nThis module provides utilities for model evaluation.")
    print("\nKey features:")
    print("- Comprehensive metrics calculation")
    print("- Prediction visualization")
    print("- Residual analysis")
    print("- Feature importance plots")
    print("- Model comparison plots")
    print("- Learning curve visualization")
