from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Define scaler and polynomial features
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2)

# Dummy dataset for fitting scaler and polynomial features (for demonstration)
dummy_data = np.array([[25, 1, 1, 1], [30, 0, 0, 0], [45, 1, 1, 1]])
scaler.fit(dummy_data)
poly.fit(scaler.transform(dummy_data))

# Define a simple model (logistic regression)
best_model = LogisticRegression()
X_dummy = poly.transform(scaler.transform(dummy_data))
y_dummy = np.array([0, 1, 0])  # Sample target values
best_model.fit(X_dummy, y_dummy)

# Initialize Label Encoder
le = LabelEncoder()

# Function to preprocess user input
def preprocess_user_input(input_data):
    df = pd.DataFrame(input_data)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

# HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autism Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        h1 {
            text-align: center;
            color: black;
            margin-bottom: 20px;
        }

        form {
            display: flex;
            flex-direction: row;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            max-width: 900px;
            width: 100%;
        }

        .form-column {
            display: flex;
            flex-direction: column;
            width: 50%;
            margin-right: 20px;
        }

        .form-column:last-child {
            margin-right: 0;
        }

        .form-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        label {
            font-size: 14px;
            color: #555;
            margin-right: 10px;
            width: 50%;
        }

        input, select {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            width: 50%;
            font-size: 14px;
        }

        button {
            padding: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 0;
        }

        button:hover {
            background-color: #0056b3;
        }

        #result {
            text-align: center;
            margin-top: 20px;
            font-size: 16px;
            color: #333;
        }

        .container {
            width: 100%;
            margin: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Autism Prediction</h1>
        <form id="predictionForm">
            <!-- First Column -->
            <div class="form-column">
                <div class="form-item">
                    <label for="age">Age:</label>
                    <input type="number" name="age" min="0" max="100" required>
                </div>
                <div class="form-item">
                    <label for="gender">Gender:</label>
                    <select name="gender" required>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="ethnicity">Ethnicity:</label>
                    <select name="ethnicity" required>
                        <option value="White">White</option>
                        <option value="Asian">Asian</option>
                        <option value="Black">Black</option>
                        <option value="Hispanic">Hispanic</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="family_history_asd">Family history of ASD:</label>
                    <select name="family_history_asd" required>
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="pre_existing_conditions">Pre-existing conditions:</label>
                    <select name="pre_existing_conditions" required>
                        <option value="none">None</option>
                        <option value="diabetes">Diabetes</option>
                        <option value="hypertension">Hypertension</option>
                        <option value="asthma">Asthma</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="developmental_milestones">Developmental milestones:</label>
                    <select name="developmental_milestones" required>
                        <option value="normal">Normal</option>
                        <option value="delayed">Delayed</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="aq_score">AQ Score:</label>
                    <input type="number" name="aq_score" min="0" max="100" required>
                </div>
                <div class="form-item">
                    <label for="social_interactions">Social interactions:</label>
                    <select name="social_interactions" required>
                        <option value="good">Good</option>
                        <option value="poor">Poor</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="communication_skills">Communication skills:</label>
                    <select name="communication_skills" required>
                        <option value="good">Good</option>
                        <option value="poor">Poor</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="genetic_markers">Genetic markers:</label>
                    <select name="genetic_markers" required>
                        <option value="negative">Negative</option>
                        <option value="positive">Positive</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="genomic_test_results">Genomic test results:</label>
                    <select name="genomic_test_results" required>
                        <option value="negative">Negative</option>
                        <option value="positive">Positive</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="environmental_toxins">Environmental toxins:</label>
                    <select name="environmental_toxins" required>
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="prenatal_factors">Prenatal factors:</label>
                    <select name="prenatal_factors" required>
                        <option value="normal">Normal</option>
                        <option value="complicated">Complicated</option>
                    </select>
                </div>
            </div>

            <!-- Second Column -->
            <div class="form-column">
                <div class="form-item">
                    <label for="psychological_evaluations">Psychological evaluations:</label>
                    <select name="psychological_evaluations" required>
                        <option value="normal">Normal</option>
                        <option value="abnormal">Abnormal</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="neurological_assessments">Neurological assessments:</label>
                    <select name="neurological_assessments" required>
                        <option value="normal">Normal</option>
                        <option value="abnormal">Abnormal</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="language_development">Language development:</label>
                    <select name="language_development" required>
                        <option value="normal">Normal</option>
                        <option value="delayed">Delayed</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="motor_skills">Motor skills:</label>
                    <select name="motor_skills" required>
                        <option value="normal">Normal</option>
                        <option value="delayed">Delayed</option>
                    </select>
                </div>
                <div class="form-item">
                    <label for="behavioral_assessment_score">Behavioral assessment score:</label>
                    <input type="number" name="behavioral_assessment_score" required>
                </div>
                <div class="form-item">
                    <label for="screen_time">Daily screen time (hours):</label>
                    <input type="number" name="screen_time" required>
                </div>
                <div class="form-item">
                    <label for="parenting_style">Parenting style:</label>
                    <input type="text" name="parenting_style" required>
                </div>
                <div class="form-item">
                    <label for="sleep_quality">Sleep quality:</label>
                    <input type="text" name="sleep_quality" required>
                </div>
                <div class="form-item">
                    <label for="dietary_habits">Dietary habits:</label>
                    <input type="text" name="dietary_habits" required>
                </div>
                <div class="form-item">
                    <label for="physical_activity">Physical activity level:</label>
                    <input type="text" name="physical_activity" required>
                </div>
                <div class="form-item">
                    <label for="social_engagement">Social engagement:</label>
                    <input type="text" name="social_engagement" required>
                </div>
                <div class="form-item">
                    <label for="emotional_regulation">Emotional regulation:</label>
                    <input type="text" name="emotional_regulation" required>
                </div>
                <div class="form-item">
                    <label for="technology_use">Technology use:</label>
                    <input type="text" name="technology_use" required>
                </div>
            </div>
        </form>
        </div>

        <button type="submit" form="predictionForm">Submit</button>
        <div id="result"></div>
    

    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(event) {
            event.preventDefault();  // Prevent the default form submission

            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => { data[key] = value; });

            // Send a POST request to the /predict route
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Display the prediction result
                document.getElementById('result').innerText = 'Prediction: ' + data.prediction;
            })
            .catch(error => {
                // Handle errors
                document.getElementById('result').innerText = 'Error: ' + error.message;
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = preprocess_user_input([data])
        input_scaled = scaler.transform(input_data)
        input_poly = poly.transform(input_scaled)
        prediction = best_model.predict(input_poly)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

