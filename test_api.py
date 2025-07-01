import requests

url = 'http://127.0.0.1:5000/predict'

# Sample data based on student-mat.csv (modify only if needed)
data = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 4,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "yes",
    "famsup": "no",
    "paid": "no",
    "activities": "yes",
    "internet": "yes",
    "romantic": "no",
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 2,
    "G1": 14,
    "G2": 15
}

response = requests.post(url, json=data)
print("✅ Response:", response.json())
