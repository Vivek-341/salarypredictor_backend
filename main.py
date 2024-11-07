# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from d2 import model, encoders
# import pandas as pd

# app = FastAPI()

# # Request model
# class SalaryPredictionRequest(BaseModel):
#     Position: str
#     ExperienceYears: float
#     Degree: str
#     Field: str
#     Certification: str
#     ResearchPublications: float
#     University: str
#     Location: str

# @app.post("/predict_salary")
# async def predict_salary(request: SalaryPredictionRequest):
#     try:
#         input_data = {
#             "Position": [request.Position],
#             "ExperienceYears": [request.ExperienceYears],
#             "Degree": [request.Degree],
#             "Field": [request.Field],
#             "Certification": [request.Certification],
#             "ResearchPublications": [request.ResearchPublications],
#             "University": [request.University],
#             "Location": [request.Location],
#         }

#         # Encode input
#         for col, encoder in encoders.items():
#             if col in input_data:
#                 input_data[col] = encoder.transform(input_data[col])

#         df_input = pd.DataFrame(input_data)
#         predicted_salary = model.predict(df_input)[0]
#         return {"predicted_salary": predicted_salary}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))








# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import LabelEncoder
# from typing import List, Dict
# import numpy as np

# app = FastAPI()

# # Load and preprocess data
# data = pd.read_csv('university_employee_salaries.csv')  # Use your actual file path here
# categorical_columns = ['Position', 'Degree', 'Field', 'Certification', 'University', 'Location']
# encoders = {}

# # Encode categorical features
# for col in categorical_columns:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])
#     encoders[col] = le

# # Prepare feature matrix and target variable
# X = data[['Position', 'ExperienceYears', 'Degree', 'Field', 'Certification', 'ResearchPublications', 'University', 'Location']]
# y = data['Salary']

# # Split the dataset and train the model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train, y_train)

# # Input data model for API
# class SalaryPredictionRequest(BaseModel):
#     Position: str
#     ExperienceYears: float
#     Degree: str
#     Field: str
#     Certification: str
#     ResearchPublications: float
#     University: str
#     Location: str

# # POST endpoint for prediction
# @app.post("/predict_salary")
# async def predict_salary(request: SalaryPredictionRequest):
#     # Convert input data to DataFrame for model prediction
#     input_data = {
#         "Position": [request.Position],
#         "ExperienceYears": [request.ExperienceYears],
#         "Degree": [request.Degree],
#         "Field": [request.Field],
#         "Certification": [request.Certification],
#         "ResearchPublications": [request.ResearchPublications],
#         "University": [request.University],
#         "Location": [request.Location],
#     }

#     # Encode the input data
#     try:
#         for col in categorical_columns:
#             input_data[col] = encoders[col].transform(input_data[col])
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")

#     df_input = pd.DataFrame(input_data)
#     predicted_salary = model.predict(df_input)[0]
#     return {"predicted_salary": predicted_salary}

# # GET endpoint for general model information (e.g., model accuracy)
# @app.get("/model_info")
# async def model_info():
#     # Evaluate model accuracy on test set
#     y_pred_test = model.predict(X_test)
#     accuracy = 1 - np.mean(np.abs((y_test - y_pred_test) / y_test))
#     return {
#         "model_accuracy": f"{accuracy:.2%}",
#         "feature_importances": dict(zip(X.columns, model.feature_importances_))
#     }





# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import LabelEncoder
# import numpy as np

# app = FastAPI()

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:5500"],  # Allow only your frontend origin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load and preprocess data
# data = pd.read_csv('university_employee_salaries.csv')  # Use your actual file path here
# categorical_columns = ['Position', 'Degree', 'Field', 'Certification', 'University', 'Location']
# encoders = {}

# # Encode categorical features
# for col in categorical_columns:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])
#     encoders[col] = le

# # Prepare feature matrix and target variable
# X = data[['Position', 'ExperienceYears', 'Degree', 'Field', 'Certification', 'ResearchPublications', 'University', 'Location']]
# y = data['Salary']

# # Split the dataset and train the model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train, y_train)

# # Input data model for API
# class SalaryPredictionRequest(BaseModel):
#     Position: str
#     ExperienceYears: float
#     Degree: str
#     Field: str
#     Certification: str
#     ResearchPublications: float
#     University: str
#     Location: str

# # POST endpoint for prediction
# @app.post("/predict_salary")
# async def predict_salary(request: SalaryPredictionRequest):
#     # Convert input data to DataFrame for model prediction
#     input_data = {
#         "Position": [request.Position],
#         "ExperienceYears": [request.ExperienceYears],
#         "Degree": [request.Degree],
#         "Field": [request.Field],
#         "Certification": [request.Certification],
#         "ResearchPublications": [request.ResearchPublications],
#         "University": [request.University],
#         "Location": [request.Location],
#     }

#     # Encode the input data
#     try:
#         for col in categorical_columns:
#             input_data[col] = encoders[col].transform(input_data[col])
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")

#     df_input = pd.DataFrame(input_data)
#     predicted_salary = model.predict(df_input)[0]
#     return {"predicted_salary": predicted_salary}

# # GET endpoint for general model information (e.g., model accuracy)
# @app.get("/predict_salary")
# async def model_info():
#     # Evaluate model accuracy on test set
#     y_pred_test = model.predict(X_test)
#     accuracy = 1 - np.mean(np.abs((y_test - y_pred_test) / y_test))
#     return {
#         "model_accuracy": f"{accuracy:.2%}",
#         "feature_importances": dict(zip(X.columns, model.feature_importances_))
#     }










from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Allow only your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and preprocess data
data = pd.read_csv('university_employee_salaries.csv')  # Use your actual file path here
categorical_columns = ['Position', 'Degree', 'Field', 'Certification', 'University', 'Location']
encoders = {}

# Encode categorical features
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Prepare feature matrix and target variable
X = data[['Position', 'ExperienceYears', 'Degree', 'Field', 'Certification', 'ResearchPublications', 'University', 'Location']]
y = data['Salary']

# Split the dataset and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Calculate model accuracy on the test set
y_pred_test = model.predict(X_test)
accuracy = 1 - np.mean(np.abs((y_test - y_pred_test) / y_test))

# Input data model for API
class SalaryPredictionRequest(BaseModel):
    Position: str
    ExperienceYears: float
    Degree: str
    Field: str
    Certification: str
    ResearchPublications: float
    University: str
    Location: str

# POST endpoint for prediction with accuracy
@app.post("/predict_salary")
async def predict_salary(request: SalaryPredictionRequest):
    # Convert input data to DataFrame for model prediction
    input_data = {
        "Position": [request.Position],
        "ExperienceYears": [request.ExperienceYears],
        "Degree": [request.Degree],
        "Field": [request.Field],
        "Certification": [request.Certification],
        "ResearchPublications": [request.ResearchPublications],
        "University": [request.University],
        "Location": [request.Location],
    }

    # Encode the input data
    try:
        for col in categorical_columns:
            input_data[col] = encoders[col].transform(input_data[col])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")

    df_input = pd.DataFrame(input_data)
    predicted_salary = model.predict(df_input)[0]
    
    # Return both predicted salary and model accuracy
    return {
        "predicted_salary": predicted_salary,
        "model_accuracy": f"{accuracy:.2%}"  # Format as a percentage
    }
