# Dataset Directory

## Downloading the Ames Housing Dataset

This project uses the Ames Housing Dataset from Kaggle's "House Prices: Advanced Regression Techniques" competition.

### Instructions:

1. Visit the Kaggle competition page:
   https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

2. Download the following files:
   - `train.csv` - Training dataset with SalePrice
   - `test.csv` - Test dataset for predictions
   - `data_description.txt` - Detailed description of all features

3. Place the downloaded files in this directory (`data/`)

### Alternative: Direct Download (requires Kaggle API)

If you have the Kaggle API set up:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/
```

### Dataset Overview

The Ames Housing dataset contains:
- **Train set**: ~1460 observations
- **Test set**: ~1459 observations  
- **Features**: 79 explanatory variables + 1 target (SalePrice)
- **Feature types**: Numerical and categorical

### Feature Categories

1. **Lot/Land**: LotArea, LotShape, LotConfig, etc.
2. **Location**: Neighborhood, Condition1, Condition2
3. **Dwelling**: MSSubClass, MSZoning, BldgType, HouseStyle
4. **Quality/Condition**: OverallQual, OverallCond, ExterQual, etc.
5. **Size**: GrLivArea, TotalBsmtSF, GarageArea, etc.
6. **Rooms**: BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd
7. **Year Built/Remodeled**: YearBuilt, YearRemodAdd, GarageYrBlt
8. **Garage**: GarageType, GarageCars, GarageArea, etc.
9. **Basement**: BsmtQual, BsmtCond, BsmtFinType1, etc.
10. **Sale**: SaleType, SaleCondition, MoSold, YrSold

### Target Variable
- **SalePrice**: The property's sale price in dollars (to be predicted)

## Note
The CSV files are excluded from git tracking via `.gitignore` to keep the repository size small.
