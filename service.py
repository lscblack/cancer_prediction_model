from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI
import joblib

# Load model
model = joblib.load('logistic_regression_model.joblib')
class_names = np.array(["Malignant", "Benign"])

# FastAPI app
app = FastAPI()

# Define the feature model with all columns
class FeatureInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

# Prediction endpoint
@app.post("/predict")
async def predict(features: FeatureInput):
    feature_array = np.array([[features.radius_mean, features.texture_mean, features.perimeter_mean,
                               features.area_mean, features.smoothness_mean, features.compactness_mean,
                               features.concavity_mean, features.concave_points_mean, features.symmetry_mean,
                               features.fractal_dimension_mean, features.radius_se, features.texture_se,
                               features.perimeter_se, features.area_se, features.smoothness_se,
                               features.compactness_se, features.concavity_se, features.concave_points_se,
                               features.symmetry_se, features.fractal_dimension_se, features.radius_worst,
                               features.texture_worst, features.perimeter_worst, features.area_worst,
                               features.smoothness_worst, features.compactness_worst, features.concavity_worst,
                               features.concave_points_worst, features.symmetry_worst, features.fractal_dimension_worst]])
    
    prediction = model.predict(feature_array)
    predicted_class = class_names[prediction]
    
    return {"prediction": predicted_class[0]}
