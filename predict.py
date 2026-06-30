"""
predict.py
-----------
Loads the trained regression model and predicts a student's exam score
(a number, e.g. out of 100) from attendance + whatever subject marks the
model was trained on.

Usage:
    python predict.py
"""

import json
import joblib
import pandas as pd


def load_model():
    reg = joblib.load("score_regressor.pkl")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    return reg, metadata


def predict(reg, feature_cols, values):
    X = pd.DataFrame([values], columns=feature_cols)
    predicted_score = reg.predict(X)[0]
    return round(predicted_score, 1)


def score_comment(score):
    if score >= 85:
        return "Excellent — likely to score in the top range."
    elif score >= 70:
        return "Good — solid performance expected."
    elif score >= 50:
        return "Average — there's room to improve."
    else:
        return "At risk — additional support may help before the exam."


def main():
    reg, metadata = load_model()
    feature_cols = metadata["feature_cols"]
    target_label = metadata["target_column"].replace("_marks", "").replace("_", " ").title()
    mae = metadata["mae"]

    print("=== Exam Score Predictor ===")
    print(f"Predicting: {target_label}")
    print(f"Typical prediction error: +/- {mae} marks\n")

    values = {}
    for col in feature_cols:
        if col == "attendance":
            prompt = "Attendance percentage: "
        else:
            subject_name = col.replace("_marks", "").replace("_", " ").title()
            prompt = f"{subject_name} marks: "
        values[col] = float(input(prompt).strip())

    predicted_score = predict(reg, feature_cols, values)
    low = round(max(predicted_score - mae, 0), 1)
    high = round(min(predicted_score + mae, 100), 1)

    print(f"\nPredicted {target_label} score: {predicted_score}")
    print(f"Likely range: {low} - {high}")
    print(score_comment(predicted_score))


if __name__ == "__main__":
    main()
