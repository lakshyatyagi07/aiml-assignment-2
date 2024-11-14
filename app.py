from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler with local paths using joblib
model_path = os.path.join(os.getcwd(), 'diabetes_model.pkl')
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Home route to display form
@app.route('/')
def home():
    home_html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Diabetes Prediction</title>
    </head>
    <body>
        <h1>Diabetes Prediction</h1>
        <form action="/predict" method="post">
            <label>Pregnancies: <input type="number" name="pregnancies" step="any" required></label><br>
            <label>Glucose: <input type="number" name="glucose" step="any" required></label><br>
            <label>Blood Pressure: <input type="number" name="blood_pressure" step="any" required></label><br>
            <label>Skin Thickness: <input type="number" name="skin_thickness" step="any" required></label><br>
            <label>Insulin: <input type="number" name="insulin" step="any" required></label><br>
            <label>BMI: <input type="number" name="bmi" step="any" required></label><br>
            <label>Diabetes Pedigree Function: <input type="number" name="diabetes_pedigree_function" step="any" required></label><br>
            <label>Age: <input type="number" name="age" required></label><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    '''
    return render_template_string(home_html)

# Prediction route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = [float(request.form[key]) for key in request.form.keys()]
        
        # Reshape and scale input data
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected"
    except Exception as e:
        result = f"An error occurred: {e}"

    result_html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Prediction Result</title>
    </head>
    <body>
        <h1>Prediction Result</h1>
        <p>{result}</p>
        <a href="/">Try Again</a>
    </body>
    </html>
    '''
    return render_template_string(result_html)

if __name__ == '__main__':
    app.run(debug=True)
