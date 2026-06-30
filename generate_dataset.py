"""
generate_dataset.py
--------------------
Creates a synthetic dataset with attendance + any number of subject marks,
and saves it as student_data.csv. Replace this with your real dataset —
the rest of the project automatically detects whatever subject columns
are present (anything ending in "_marks"), so it doesn't matter how many
subjects or what they're called.

To use your own data instead of this generator:
  - Save a CSV with an "attendance" column and one or more "<subject>_marks"
    columns (e.g. "physics_marks", "history_marks", "art_marks", ...).
  - Skip running this script; just put your CSV at student_data.csv.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

# ---------------------------------------------------------------------
# Define any subjects you want here — add, remove, or rename freely.
# ---------------------------------------------------------------------
SUBJECTS = ["physics", "chemistry", "biology", "history", "computer_science"]

attendance = np.round(np.random.uniform(40, 100, N), 1)

data = {"attendance": attendance}
marks_matrix = []

for subject in SUBJECTS:
    base = np.random.uniform(20, 40)       # random base difficulty per subject
    slope = np.random.uniform(0.4, 0.65)   # how much attendance helps
    noise_sd = np.random.uniform(7, 11)
    marks = np.clip(base + attendance * slope + np.random.normal(0, noise_sd, N), 0, 100)
    marks = np.round(marks, 1)
    data[f"{subject}_marks"] = marks
    marks_matrix.append(marks)

marks_matrix = np.array(marks_matrix)
average = np.round(marks_matrix.mean(axis=0), 1)


def categorize(avg):
    if avg >= 85:
        return "Excellent"
    elif avg >= 70:
        return "Good"
    elif avg >= 50:
        return "Average"
    else:
        return "Needs Improvement"


data["average_marks"] = average
data["performance"] = [categorize(a) for a in average]
data["result"] = ["Pass" if a >= 40 else "Fail" for a in average]

df = pd.DataFrame(data)
df.to_csv("student_data.csv", index=False)

print(f"Generated student_data.csv with {len(df)} rows and subjects: {', '.join(SUBJECTS)}")
print(df.head())
