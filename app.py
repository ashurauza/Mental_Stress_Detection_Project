from flask import Flask, request, render_template, redirect, url_for
from pymongo import MongoClient
import joblib
import pandas as pd

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['StressPredictionDB']
collection = db['UserInputs']

# Load the trained model
try:
    model = joblib.load('model_rf.pkl')  # Ensure the model file is in the correct path
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect form data for prediction (only include necessary fields for prediction)
            prediction_data = {
                "anxiety_level": int(request.form["anxiety_level"]),
                "self_esteem": int(request.form["self_esteem"]),
                "mental_health_history": int(request.form["mental_health_history"]),
                "depression": int(request.form["depression"]),
                "headache": int(request.form["headache"]),
                "blood_pressure": int(request.form["blood_pressure"]),
                "sleep_quality": int(request.form["sleep_quality"]),
                "breathing_problem": int(request.form["breathing_problem"]),
                "noise_level": int(request.form["noise_level"]),
                "living_conditions": int(request.form["living_conditions"]),
                "safety": int(request.form["safety"]),
                "basic_needs": int(request.form["basic_needs"]),
                "academic_performance": int(request.form["academic_performance"]),
                "study_load": int(request.form["study_load"]),
                "teacher_student_relationship": int(request.form["teacher_student_relationship"]),
                "future_career_concerns": int(request.form["future_career_concerns"]),
                "social_support": int(request.form["social_support"]),
                "peer_pressure": int(request.form["peer_pressure"]),
                "extracurricular_activities": int(request.form["extracurricular_activities"]),
                "bullying": int(request.form["bullying"]),
            }
            
            # Collect the additional data fields to store in MongoDB (but not used in prediction)
            user_data = {
                "age": int(request.form["age"]),
                "gender": request.form["gender"],
                "education_level": request.form["education_level"],
                "graduation_year": int(request.form["graduation_year"]),
                "field_of_study": request.form["field_of_study"],
                "branch": request.form["branch"],
                "college_name": request.form["college_name"],
                "cgpa": float(request.form["cgpa"]),
                "yearly_income": int(request.form["yearly_income"]),
                "diet_type": request.form["diet_type"],
            }

            # Prepare data for prediction
            input_features = pd.DataFrame([list(prediction_data.values())], columns=prediction_data.keys())
            
            # Predict stress level
            if model:
                predicted_stress_level = int(model.predict(input_features)[0])
                prediction_data["predicted_stress_level"] = predicted_stress_level
            else:
                prediction_data["predicted_stress_level"] = None  # Fallback if the model is not loaded
            
            # Insert both prediction data and user data into MongoDB
            user_data.update(prediction_data)  # Combine the prediction data with the user data
            collection.insert_one(user_data)
            
            return render_template("result.html", prediction=user_data["predicted_stress_level"])
        
        except Exception as e:
            return render_template("result.html", error=f"An error occurred: {str(e)}")

    return render_template("predict.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
