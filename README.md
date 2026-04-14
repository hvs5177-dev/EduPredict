# EduPredict AI - Student Performance Prediction

A machine learning project that predicts student final marks using a deep learning ANN model built with Keras/TensorFlow.

## 📊 Project Structure

```
EduPredict_AI/
├── data/                              # Dataset folder
│   └── students_data.csv             # Generated synthetic dataset (10,000+ rows)
├── models/                            # Trained models folder
│   ├── student_performance_model.h5  # Trained ANN model
│   ├── scaler.pkl                    # StandardScaler for input normalization
│   └── feature_names.pkl             # Feature names reference
├── generate_dataset.py               # Dataset generation script
├── train_model.py                    # Model training script
├── app.py                            # Streamlit prediction app
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 📋 Dataset Features

The dataset contains 10,000+ student records with the following features:
- **age**: Student age (16-24 years)
- **study_hours**: Daily study hours (1-12)
- **attendance**: Attendance percentage (40-100%)
- **math**: Math score (0-100)
- **science**: Science score (0-100)
- **english**: English score (0-100)
- **previous**: Previous semester marks (30-95)
- **final_marks**: Target variable - final marks (0-100)

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Dataset

Run this to create the synthetic dataset:

```bash
python generate_dataset.py
```

**Output:**
- Creates `data/students_data.csv` with 10,000 rows
- Displays dataset statistics and sample rows

### Step 3: Train the Model

```bash
python train_model.py
```

**Output:**
- Trains an ANN with 4 hidden layers
- Saves model to `models/student_performance_model.h5`
- Saves scaler to `models/scaler.pkl`
- Saves feature names to `models/feature_names.pkl`
- Displays training and test metrics

**Expected Performance:**
- R² Score (Test): ~0.90+
- MAE (Test): ~3-4 marks

### Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

**Access:** Opens at `http://localhost:8501`

## 🎨 App Features

### Input Interface (Sidebar)
- Age slider (16-24 years)
- Daily study hours (1-12 hours)
- Attendance percentage (40-100%)
- Subject scores (Math, Science, English): 0-100
- Previous semester marks (30-95)

### Output Display
- **Predicted Final Marks**: AI-predicted score (0-100)
- **Pass/Fail Status**: Passes if ≥ 50, fails if < 50
- **Risk Level**:
  - 🟢 Low Risk: ≥ 80 (Excellent)
  - 🟡 Moderate Risk: 70-79 (Good)
  - 🟠 Medium Risk: 60-69 (Adequate)
  - 🔴 High Risk: 50-59 (Below Average)
  - 🔴 Critical Risk: < 50 (Poor)

### Visualizations
- Bar chart comparing subject scores with predicted final marks
- Pass threshold line overlay
- Performance summary table
- Input parameters summary

### Insights & Recommendations
- Identifies strong and weak subject areas
- Provides study hour recommendations
- Attendance improvement suggestions
- Overall performance assessment

## 🧠 Model Architecture

The ANN model consists of:

```
Input Layer: 7 features
↓
Dense(128) + ReLU + Dropout(0.2)
↓
Dense(64) + ReLU + Dropout(0.2)
↓
Dense(32) + ReLU + Dropout(0.1)
↓
Dense(16) + ReLU
↓
Output Layer: 1 (final marks)
```

**Training Configuration:**
- Optimizer: Adam (learning rate: 0.001)
- Loss: Mean Squared Error (MSE)
- Batch Size: 32
- Epochs: 100
- Validation Split: 20%
- Test Size: 20%

## ⚙️ Key Implementation Details

### 1. Data Preprocessing
- **Train-Test Split**: 80% training, 20% testing
- **Feature Scaling**: StandardScaler for normalization
- **Validation Split**: 20% of training data used for validation

### 2. Scaling Consistency
- Scaler is **fit only on training data** to prevent data leakage
- Same scaler is used for:
  - Transforming test data during training
  - Transforming input data in the Streamlit app
- This ensures predictions are on the same scale as training

### 3. Model Safeguards
- Output clipped to [0, 100] range to avoid unrealistic predictions
- Dropout layers prevent overfitting
- Validation metrics track training progress
- Test set metrics measure generalization

### 4. Feature Engineering
- Dataset generation uses realistic relationships:
  - Subject scores influence final marks
  - Study hours have strong positive correlation
  - Attendance contributes to performance
  - Previous marks indicate consistent performance

## 📈 Expected Results

After running all steps, you should see:

**generate_dataset.py:**
```
Dataset shape: (10000, 8)
Statistics showing realistic distributions
```

**train_model.py:**
```
Train R² Score: ~0.92-0.95
Test R² Score: ~0.90-0.93
Test MAE: ~3-4 marks
```

**app.py:**
- Interactive interface for single predictions
- Visualization of performance metrics
- Personalized recommendations

## 🔧 Troubleshooting

### Model not found error in app.py
- Ensure you've run `train_model.py` first
- Check that `models/` folder contains the saved files

### Import errors
- Install all dependencies: `pip install -r requirements.txt`
- Use Python 3.8 or higher

### Prediction seems unrealistic
- Input should be in reasonable ranges (age 16-24, scores 0-100)
- Scaler is automatically applied; don't pre-scale inputs

### Port 8501 already in use
```bash
streamlit run app.py --server.port=8502
```

## 📚 References

- TensorFlow/Keras: https://tensorflow.org
- Scikit-learn: https://scikit-learn.org
- Streamlit: https://streamlit.io
- Plotly: https://plotly.com

## 📝 License

This project is provided as-is for educational purposes.

---

**Created**: April 2026  
**Purpose**: Student Performance Prediction using Deep Learning  
**Status**: Production Ready ✅
