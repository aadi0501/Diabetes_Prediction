from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Initialize Flask App
app = Flask(__name__)

# Load the data into pandas dataframe
diabetes_data = pd.read_csv("https://raw.githubusercontent.com/aadi0501/Diabetes_Data/refs/heads/main/diabetes.csv")

# Prepare the data
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Flask Routes
@app.route('/')
def home():
    # Home page with the form for input
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Diabetes Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f4f4f9;
            }
            .container {
                background: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                text-align: center;
                width: 300px;
            }
            input {
                margin: 5px;
                padding: 8px;
                width: 90%;
            }
            button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Diabetes Prediction</h1>
            <form method="POST" action="/predict">
                <input type="text" name="Pregnancies" placeholder="Pregnancies" required><br>
                <input type="text" name="Glucose" placeholder="Glucose" required><br>
                <input type="text" name="BloodPressure" placeholder="Blood Pressure" required><br>
                <input type="text" name="SkinThickness" placeholder="Skin Thickness" required><br>
                <input type="text" name="Insulin" placeholder="Insulin" required><br>
                <input type="text" name="BMI" placeholder="BMI" required><br>
                <input type="text" name="DiabetesPedigreeFunction" placeholder="Diabetes Pedigree Function" required><br>
                <input type="text" name="Age" placeholder="Age" required><br>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_data = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age']),
    ]

    # Standardize the input data
    input_data_np = np.array(input_data).reshape(1, -1)
    std_input_data = scaler.transform(input_data_np)

    # Make the prediction
    prediction = classifier.predict(std_input_data)

    # Determine result
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

    # Return a simple response with the prediction
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f4f4f9;
            }}
            .container {{
                background: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                text-align: center;
                width: 300px;
            }}
            a {{
                text-decoration: none;
                color: white;
            }}
            button {{
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Prediction Result</h1>
            <p><strong>{result}</strong></p>
            <button><a href="/">Back to Home</a></button>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
