import joblib

# Load trained model
model = joblib.load("student_placement_model.pkl")

# Take student input
cgpa = float(input("Enter CGPA: "))
attendance = float(input("Enter Attendance: "))
coding_score = float(input("Enter Coding Score: "))
projects = int(input("Enter Number of Projects: "))
internship = int(input("Enter Internship (1=Yes, 0=No): "))

# Create input data
student = [[
    cgpa,
    attendance,
    coding_score,
    projects,
    internship
]]

# Make prediction
prediction = model.predict(student)[0]

if prediction == 1:
    print("Student is likely to be PLACED")
else:
    print("Student is likely to NOT BE PLACED")