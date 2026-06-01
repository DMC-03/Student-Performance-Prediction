import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("student-performance.csv")

# Drop Overall_Percentage if present
if "Overall_Percentage" in df.columns:
    df = df.drop(columns=["Overall_Percentage"])

# Encode categorical columns
le_dict = {}
for col in ["Department", "Extracurricular_Activities", "Internet_Access_at_Home", "Grade"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Add slight noise to numeric columns for generalization
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove("Grade")
for col in num_cols:
    df[col] += np.random.normal(0, 2, df.shape[0])

# Clip values to valid ranges
for col in ["Midterm_Score", "Assignments_Avg", "Projects_Score", "Quizzes_Avg", "Attendance (%)"]:
    df[col] = df[col].clip(0, 100)
df["Sleep_Hours_per_Night"] = df["Sleep_Hours_per_Night"].clip(0, 12)
df["Study_Hours_per_Week"] = df["Study_Hours_per_Week"].clip(0, 40)
df["Phone_Screen_Time (hrs/day)"] = df["Phone_Screen_Time (hrs/day)"].clip(0, 10)

# Split dataset
X = df.drop("Grade", axis=1)
y = df["Grade"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y, shuffle=True
)

# Train Decision Tree model
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    min_samples_leaf=8,
    min_samples_split=15,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
cv_scores = cross_val_score(model, X, y, cv=5)
cv_accuracy = round(cv_scores.mean() * 100, 2)

print(f"Test Accuracy: {accuracy}%")
print(f"Cross-Validation Accuracy: {cv_accuracy}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save model and encoders
joblib.dump(model, "student_grade_model.pkl")
joblib.dump(le_dict, "label_encoders.pkl")
print("Model and encoders saved successfully.")
