import joblib
import pandas as pd


def test_model_exists():
    model = joblib.load("student_placement_model.pkl")
    assert model is not None


def test_model_prediction():
    model = joblib.load("student_placement_model.pkl")

    student = pd.DataFrame([{
        "CGPA": 8.5,
        "Attendance": 92,
        "CodingScore": 85,
        "Projects": 3,
        "Internship": 1
    }])

    prediction = model.predict(student)

    assert prediction[0] in [0, 1]