"""
Generate a realistic synthetic dataset for student performance prediction.
"""

import pandas as pd
import numpy as np
import os

# Seed
np.random.seed(42)

n_samples = 10000
print(f"Generating {n_samples} student records...")

# -----------------------------
# Generate Features
# -----------------------------
data = {
    'age': np.random.randint(16, 25, n_samples),
    'study_hours': np.random.uniform(1, 12, n_samples),
    'attendance': np.random.uniform(40, 100, n_samples),
    'math': np.random.randint(20, 100, n_samples),
    'science': np.random.randint(20, 100, n_samples),
    'english': np.random.randint(20, 100, n_samples),
    'previous': np.random.randint(30, 95, n_samples),
}

df = pd.DataFrame(data)

# -----------------------------
# Normalize features
# -----------------------------
study_norm = df['study_hours'] / 12       # 0–1
attendance_norm = df['attendance'] / 100  # 0–1

# -----------------------------
# REALISTIC FINAL MARKS LOGIC
# -----------------------------
df['final_marks'] = (
    0.30 * df['math'] +
    0.30 * df['science'] +
    0.20 * df['english'] +
    0.10 * df['previous'] +
    10 * study_norm +       # max ~10 contribution
    5 * attendance_norm     # max ~5 contribution
)

# -----------------------------
# Add noise (realistic variation)
# -----------------------------
df['final_marks'] += np.random.normal(0, 5, n_samples)

# Clip between 0–100
df['final_marks'] = np.clip(df['final_marks'], 0, 100)

# Round values
df['study_hours'] = df['study_hours'].round(2)
df['attendance'] = df['attendance'].round(2)
df['final_marks'] = df['final_marks'].round(2)

# -----------------------------
# Save dataset
# -----------------------------
os.makedirs('data', exist_ok=True)
output_path = 'data/student_data.csv'

df.to_csv(output_path, index=False)

# -----------------------------
# Logs
# -----------------------------
print("\n✅ Dataset generated successfully!")
print(f"Saved to: {output_path}")
print(f"Shape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())

print("\nStatistics:")
print(df.describe())