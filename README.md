# House Price Prediction - Ames Housing Dataset

## Project Overview
This project implements a comprehensive data analysis and machine learning solution for predicting house prices based on the Ames Housing Dataset. The dataset contains 79 explanatory variables describing various aspects of residential homes in Ames, Iowa.

## Tasks

### Task 1: Complete Data Analysis Report
- Comprehensive exploratory data analysis (EDA)
- Statistical analysis of all 79 features
- Visualizations of key relationships
- Data quality assessment and missing value analysis
- Summary insights and findings

### Task 2: Machine Learning Model
#### a) Price Prediction Algorithm
- Robust machine learning models including:
  - Random Forest Regressor
  - Gradient Boosting (XGBoost, LightGBM)
  - Linear Regression (baseline)
- Advanced feature engineering
- Hyperparameter tuning
- Model evaluation and comparison

#### b) Feature Relationship Analysis
- Correlation analysis between features
- Feature importance ranking
- Price variation analysis based on house features
- Visualizations of key relationships

### Task 3: Customer Recommendations
- Recommendation system for house buying decisions
- Filtering based on area, price, and other requirements
- Personalized suggestions based on customer preferences

## Project Structure
```
.
├── data/                          # Data directory (download dataset here)
│   └── README.md                  # Instructions for downloading data
├── notebooks/
│   ├── 01_data_analysis.ipynb    # Task 1: EDA and analysis
│   ├── 02_model_training.ipynb   # Task 2: Model development
│   └── 03_recommendations.ipynb  # Task 3: Recommendation system
├── src/
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── feature_engineering.py    # Feature creation and selection
│   ├── model_training.py         # Model training pipeline
│   ├── model_evaluation.py       # Model evaluation utilities
│   └── recommendation_system.py  # House recommendation engine
├── models/                        # Saved model files
├── outputs/                       # Generated reports and plots
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Dataset
The Ames Housing Dataset contains information about house sales in Ames, Iowa. It includes:
- 79 explanatory variables
- Features covering house characteristics, location, quality, condition, and more
- Target variable: SalePrice

### Downloading the Dataset
The dataset can be obtained from:
- [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

Download `train.csv` and `test.csv` and place them in the `data/` directory.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Purandhar-Srikanth/Python_Practise.git
cd Python_Practise
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset and place it in the `data/` directory.

## Usage

### 1. Data Analysis
Run the data analysis notebook:
```bash
jupyter notebook notebooks/01_data_analysis.ipynb
```

### 2. Train Models
```bash
python src/model_training.py
```

### 3. Make Predictions
```bash
python src/predict.py --input data/test.csv --output predictions.csv
```

### 4. Get House Recommendations
```bash
python src/recommendation_system.py --area 1500 --max-price 200000
```

## Key Features

### Data Analysis
- Missing value analysis and imputation strategies
- Distribution analysis of numerical and categorical features
- Outlier detection and handling
- Feature correlation analysis

### Feature Engineering
- Creation of new features from existing ones
- Polynomial features for key variables
- Interaction terms between important features
- Feature scaling and normalization

### Machine Learning Models
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting for high accuracy
- **LightGBM**: Fast gradient boosting framework
- Cross-validation for robust evaluation
- Hyperparameter tuning using GridSearchCV

### Model Evaluation
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
- Cross-validation scores

## Results
Detailed results, visualizations, and model performance metrics are available in the notebooks and output reports.

## Contributing
Feel free to open issues or submit pull requests for improvements.

## License
This project is for educational purposes.

## Acknowledgments
- Dataset: Ames Housing Dataset (Dean De Cock)
- Competition: Kaggle House Prices: Advanced Regression Techniques
