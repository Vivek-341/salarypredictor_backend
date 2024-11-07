import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load and preprocess data (same as before)
data = pd.read_csv('university_employee_salaries.csv')
le = LabelEncoder()
categorical_columns = ['Position', 'Degree', 'Field', 'Certification', 'University', 'Location']
encoders = {}
for col in categorical_columns:
    encoders[col] = LabelEncoder().fit(data[col])
    data[col] = encoders[col].transform(data[col])

X = data[['Position', 'ExperienceYears', 'Degree', 'Field', 'Certification', 'ResearchPublications', 'University', 'Location']]
y = data['Salary']

# Train the model (same as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

def get_user_input():
    # inputs = {}
    # inputs['Position'] = input("Enter Position (e.g., Lecturer, Professor, Research Scientist): ")
    # inputs['ExperienceYears'] = float(input("Enter Experience Years: "))
    # inputs['Degree'] = input("Enter Degree (e.g., Bachelor's, Master's, PhD): ")
    # inputs['Field'] = input("Enter Field (e.g., Science, Medicine, Business): ")
    # inputs['Certification'] = input("Enter Certification (None, Certified, Advanced Certified): ")
    # inputs['ResearchPublications'] = float(input("Enter Research P University: "))
    # inputs['University'] = input("Enter University (e.g., University A, University B): ")
    # inputs['Location'] = input("Enter Location (Urban, Suburban, Rural): ")

    encoded_input = []
    for col in X.columns:
        if col in encoders:
            try:
                encoded_value = encoders[col].transform([inputs[col]])[0]
            except ValueError:
                print(f"Warning: '{inputs[col]}' is not in training data for {col}. Using default value.")
                encoded_value = len(encoders[col].classes_)  # Assign a new category
        else:
            encoded_value = inputs[col]
        encoded_input.append(encoded_value)

    return encoded_input

# Get user input and predict salary
user_input = get_user_input()
predicted_salary = model.predict([user_input])[0]
print(f"\nPredicted Salary: ${predicted_salary:.2f}")

# Calculate and display accuracy (same as before)
y_pred_all = model.predict(X)
accuracy = 1 - np.mean(np.abs((y - y_pred_all) / y))
print(f"\nModel Accuracy: {accuracy:.2%}")

# Display feature importance (same as before)
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)