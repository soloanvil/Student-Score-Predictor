import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\asus\Desktop\Students Score Predictor\archive 2\student_exam_data.csv")

# Generate Estimated Exam Score
np.random.seed(42)
df["Estimated Exam Score"] = df["Pass/Fail"].apply(lambda x: np.random.randint(50, 100) if x == 1 else np.random.randint(0, 50))

# Define features and targets
X = df[['Study Hours', 'Previous Exam Score']]
y_score = df['Estimated Exam Score']
y_pass = df['Pass/Fail']

# Split data
X_train, X_test, y_score_train, y_score_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
X_train_pass, X_test_pass, y_pass_train, y_pass_test = train_test_split(X, y_pass, test_size=0.2, random_state=42)

# Train models
score_model = LinearRegression()
score_model.fit(X_train, y_score_train)

pass_model = LogisticRegression()
pass_model.fit(X_train_pass, y_pass_train)

# Save models
joblib.dump(score_model, "student_score_model.pkl")
joblib.dump(pass_model, "student_pass_model.pkl")

print("Models trained and saved successfully.")