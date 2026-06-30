"""
visualize.py
-------------
Generates data visualizations from student_data.csv, including:
  1. Bar chart of MEAN marks per subject
  2. Histogram of average marks distribution
  3. Pie chart of performance category distribution
  4. Scatter plots of attendance vs. each subject's marks (with trend line)
  5. Correlation heatmap across attendance + all subjects
  6. Box plot comparing subject mark spreads

Works with ANY subjects — it auto-detects every "<subject>_marks" column,
so it doesn't matter how many subjects are in your dataset or what they're
called.

All charts are saved as PNG files in the current directory.

Usage:
    python visualize.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
PALETTE = "viridis"

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("student_data.csv")

subject_cols = [c for c in df.columns if c.endswith("_marks") and c != "average_marks"]
subject_names = [c.replace("_marks", "").replace("_", " ").title() for c in subject_cols]

print(f"Loaded {len(df)} students | Subjects: {', '.join(subject_names)}\n")

# ---------------------------------------------------------------------
# 2. Bar chart — MEAN marks per subject
# ---------------------------------------------------------------------
means = df[subject_cols].mean().round(1)

plt.figure(figsize=(8, 5))
bars = plt.bar(subject_names, means.values, color=sns.color_palette(PALETTE, len(subject_cols)))
plt.title("Mean Marks per Subject", fontsize=14, fontweight="bold")
plt.ylabel("Mean Marks")
plt.ylim(0, 100)
for bar, value in zip(bars, means.values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("mean_marks_per_subject.png", dpi=150)
plt.close()
print("Saved: mean_marks_per_subject.png")

# ---------------------------------------------------------------------
# 3. Histogram — distribution of average marks
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["average_marks"], bins=20, kde=True, color="#4C72B0")
plt.axvline(df["average_marks"].mean(), color="red", linestyle="--",
            label=f"Mean = {df['average_marks'].mean():.1f}")
plt.title("Distribution of Average Marks", fontsize=14, fontweight="bold")
plt.xlabel("Average Marks")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.savefig("average_marks_distribution.png", dpi=150)
plt.close()
print("Saved: average_marks_distribution.png")

# ---------------------------------------------------------------------
# 4. Pie chart — performance category breakdown
# ---------------------------------------------------------------------
if "performance" in df.columns:
    counts = df["performance"].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
        colors=sns.color_palette(PALETTE, len(counts)),
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    plt.title("Performance Category Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("performance_breakdown.png", dpi=150)
    plt.close()
    print("Saved: performance_breakdown.png")

# ---------------------------------------------------------------------
# 5. Scatter plots — attendance vs. each subject's marks
# ---------------------------------------------------------------------
n_subjects = len(subject_cols)
cols = 2
rows = (n_subjects + 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
axes = np.array(axes).reshape(-1)

for i, (col, name) in enumerate(zip(subject_cols, subject_names)):
    ax = axes[i]
    sns.regplot(
        x="attendance", y=col, data=df, ax=ax,
        scatter_kws={"alpha": 0.4, "s": 15}, line_kws={"color": "red"},
    )
    corr = df["attendance"].corr(df[col])
    ax.set_title(f"Attendance vs {name} Marks (r={corr:.2f})", fontsize=11)
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel(f"{name} Marks")

# Hide any unused subplot slots
for j in range(n_subjects, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("attendance_vs_marks.png", dpi=150)
plt.close()
print("Saved: attendance_vs_marks.png")

# ---------------------------------------------------------------------
# 6. Correlation heatmap — attendance + all subjects
# ---------------------------------------------------------------------
corr_cols = ["attendance"] + subject_cols
corr_df = df[corr_cols].copy()
corr_df.columns = ["Attendance"] + subject_names

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# ---------------------------------------------------------------------
# 7. Box plot — spread of marks per subject
# ---------------------------------------------------------------------
melted = df[subject_cols].copy()
melted.columns = subject_names
melted = melted.melt(var_name="Subject", value_name="Marks")

plt.figure(figsize=(8, 5))
sns.boxplot(x="Subject", y="Marks", hue="Subject", data=melted, palette=PALETTE, legend=False)
plt.title("Marks Spread per Subject", fontsize=14, fontweight="bold")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("marks_spread_boxplot.png", dpi=150)
plt.close()
print("Saved: marks_spread_boxplot.png")

print("\nAll visualizations generated successfully.")