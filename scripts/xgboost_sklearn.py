# Since we are predicting QB yards, we need to compute continuous values, so
# we are using regression techniques

# Boilerplate code below

import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# 1. Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the Regressor
reg_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    objective='reg:squarederror', # Default loss for regression
    random_state=42
)

# 4. Train and predict
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

# 5. Evaluate performance
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Root Mean Squared Error: {rmse:.4f}")

