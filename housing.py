# Code you have previously used to load data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]  # Create the random grid

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = 'input/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice

features = home_data.drop(labels=['LotFrontage', 'Id', 'SalePrice', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence',
                                  'MiscFeature'], axis=1).select_dtypes(['number']).columns
# Create X
X = home_data[features]
# fill null values
X = X.fillna(method='ffill')

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# First create the base model to tune
rf_model = RandomForestRegressor()

# Random search of parameters, using 5 fold cross validation,
# search across 150 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=random_grid, n_iter=150,
                               cv=5, verbose=2, random_state=42, n_jobs=-1)  # Fit the random search model

rf_random.fit(train_X, train_y)

print(rf_random.best_params_)

rf_val_predictions = rf_random.best_estimator_.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))