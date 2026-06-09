import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from clari_tree import CLARITree


def report(name, y_true, pred):
    print(f"\n{name}")
    print("MAE: ", mean_absolute_error(y_true, pred))
    print("RMSE:", mean_squared_error(y_true, pred) ** 0.5)
    print("R^2: ", r2_score(y_true, pred))


data = fetch_california_housing()
X = data.data.astype(float)
y = data.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

ref_model = XGBRegressor(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.015,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_lambda=5.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    random_state=0,
    n_jobs=-1,
)

ref_model.fit(X_train, y_train)

ref_train = ref_model.predict(X_train)
ref_test = ref_model.predict(X_test)

report("Reference XGBoost train", y_train, ref_train)
report("Reference XGBoost test", y_test, ref_test)

tree = CLARITree(
    kappa=0.01,
    depth=5,
    lambda_=0.005,
    rho=0.05,
    eta=0.50,
    n_thresholds=30,
    thresholds_strategy="quantile",
    verbose=True,
    min_leaf_node_size=50,
)

obj = tree.fit_with_reference(X_train, y_train, ref_train)
pred = tree.predict_with_reference(X_test, ref_test)

report("Three-leaf CLARITree test", y_test, pred)

print("\nObjective:", obj)
print(tree.print_tree())