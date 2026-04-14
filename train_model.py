import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import joblib
import os

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/student_data.csv")

# Features & target
X = df.drop("final_marks", axis=1)
y = df["final_marks"]

# -------------------------------
# Convert to PASS/FAIL for confusion matrix
# -------------------------------
y_class = (y >= 40).astype(int)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test, y_train_cls, y_test_cls = train_test_split(
    X, y, y_class, test_size=0.2, random_state=42
)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# -------------------------------
# Build model
# -------------------------------
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -------------------------------
# Train model (IMPORTANT: save history)
# -------------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# -------------------------------
# Save model
# -------------------------------
model.save("models/student_model.h5")

# -------------------------------
# 📈 LOSS GRAPH
# -------------------------------
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("models/loss_graph.png")
plt.close()

# -------------------------------
# 📈 MAE GRAPH
# -------------------------------
plt.figure()
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title("Training vs Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.savefig("models/mae_graph.png")
plt.close()

# -------------------------------
# 🧠 CONFUSION MATRIX
# -------------------------------
y_pred = model.predict(X_test)
y_pred_cls = (y_pred >= 40).astype(int)

cm = confusion_matrix(y_test_cls, y_pred_cls)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Pass/Fail)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("models/confusion_matrix.png")
plt.close()

# -------------------------------
# 🧠 FEATURE IMPORTANCE (FIXED)
# -------------------------------
plt.figure(figsize=(6, 5))
corr = df.corr()

sns.heatmap(corr, cmap='coolwarm')  # ❌ no annot → no memory issue

plt.title("Feature Importance (Correlation)")
plt.savefig("models/feature_importance.png")
plt.close()

# -------------------------------
# DONE
# -------------------------------
print("✅ Model trained successfully!")
print("📊 All graphs saved in models/ folder")