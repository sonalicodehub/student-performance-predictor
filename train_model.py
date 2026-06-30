"""
train_model.py
----------------
Trains a regression model that predicts a student's EXAM SCORE (a number,
not a category) from attendance + any subject marks found in
student_data.csv.

It auto-detects every column ending in "_marks" (except "average_marks")
and uses those as features, along with "attendance". The target is
"average_marks" (treated here as the overall exam score) — change
TARGET_COLUMN below if you want to predict one specific subject's score
instead of the overall average.

Run this once to train and save the model, then use predict.py.
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

TARGET_COLUMN = "computer_science_marks"  # the subject/exam score to predict
# Tip: set this to whichever subject's upcoming exam you want to forecast.
# It will be predicted from attendance + all OTHER subjects' marks.

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("student_data.csv")

# Auto-detect subject mark columns, excluding the target itself
subject_cols = [
    c for c in df.columns
    if c.endswith("_marks") and c != "average_marks" and c != TARGET_COLUMN
]

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in student_data.csv")

feature_cols = ["attendance"] + subject_cols
print(f"Predicting target: {TARGET_COLUMN}")
print(f"Using features: {feature_cols}")

X = df[feature_cols]
y = df[TARGET_COLUMN]

# ---------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------
# 3. Train regressor
# ---------------------------------------------------------------------
reg = RandomForestRegressor(n_estimators=300, random_state=42)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Exam Score Regressor ===")
print(f"Mean Absolute Error: {mae:.2f} marks")
print(f"Root Mean Squared Error: {rmse:.2f} marks")
print(f"R^2 Score: {r2:.3f}  (closer to 1.0 is better)")

cv_scores = cross_val_score(reg, X, y, cv=5, scoring="r2")
print(f"5-Fold Cross-Validated R^2: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ---------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------
print("\n=== Feature Importance ===")
for feature, importance in sorted(zip(feature_cols, reg.feature_importances_), key=lambda x: -x[1]):
    print(f"{feature:20s} {importance:.3f}")

# ---------------------------------------------------------------------
# 5. Save model + metadata
# ---------------------------------------------------------------------
joblib.dump(reg, "score_regressor.pkl")

# Store the typical prediction error (MAE) so predict.py can show a range
metadata = {
    "feature_cols": feature_cols,
    "target_column": TARGET_COLUMN,
    "mae": round(mae, 2),
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f)

print("\nModel saved: score_regressor.pkl, model_metadata.json")
