from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

app = Flask(__name__)

# Initialize label encoders and model components
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2)
best_model = LogisticRegression()

# Initialize Label Encoders for categorical features
categorical_features = ['gender', 'ethnicity', 'family_history_asd', 'pre_existing_conditions', 
                        'developmental_milestones', 'social_interactions', 'communication_skills', 
                        'genetic_markers', 'genomic_test_results', 'environmental_toxins', 
                        'prenatal_factors', 'psychological_evaluations', 'neurological_assessments',
                        'language_development', 'motor_skills']

le_dict = {feature: LabelEncoder() for feature in categorical_features}

# HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autism Prediction</title>
</head>
<body>
    <h1>Autism Prediction</h1>
    <form id="predictionForm">
        <label for="age">Age:</label>
        <input type="number" name="age" min="0" max="100" required>

        <label for="gender">Gender:</label>
        <select name="gender" required>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
        </select>

        <label for="ethnicity">Ethnicity:</label>
        <select name="ethnicity" required>
            <option value="White">White</option>
            <option value="Asian">Asian</option>
            <option value="Black">Black</option>
        </select>

        <label for="family_history_asd">Family History of ASD:</label>
        <select name="family_history_asd" required>
            <option value="yes">Yes</option>
            <option value="no">No</option>
        </select>

        <label for="pre_existing_conditions">Pre-existing Conditions:</label>
        <select name="pre_existing_conditions" required>
            <option value="none">None</option>
            <option value="diabetes">Diabetes</option>
            <option value="asthma">Asthma</option>
        </select>

        <label for="developmental_milestones">Developmental Milestones:</label>
        <select name="developmental_milestones" required>
            <option value="normal">Normal</option>
            <option value="delayed">Delayed</option>
        </select>

        <label for="aq_score">AQ Score:</label>
        <input type="number" name="aq_score" min="0" required>

        <label for="social_interactions">Social Interactions:</label>
        <select name="social_interactions" required>
            <option value="good">Good</option>
            <option value="poor">Poor</option>
        </select>

        <label for="communication_skills">Communication Skills:</label>
        <select name="communication_skills" required>
            <option value="good">Good</option>
            <option value="poor">Poor</option>
        </select>

        <label for="genetic_markers">Genetic Markers:</label>
        <select name="genetic_markers" required>
            <option value="negative">Negative</option>
            <option value="positive">Positive</option>
        </select>

        <label for="genomic_test_results">Genomic Test Results:</label>
        <select name="genomic_test_results" required>
            <option value="negative">Negative</option>
            <option value="positive">Positive</option>
        </select>

        <label for="environmental_toxins">Environmental Toxins:</label>
        <select name="environmental_toxins" required>
            <option value="no">No</option>
            <option value="yes">Yes</option>
        </select>

        <label for="prenatal_factors">Prenatal Factors:</label>
        <select name="prenatal_factors" required>
            <option value="normal">Normal</option>
            <option value="complicated">Complicated</option>
        </select>

        <label for="psychological_evaluations">Psychological Evaluations:</label>
        <select name="psychological_evaluations" required>
            <option value="normal">Normal</option>
            <option value="abnormal">Abnormal</option>
        </select>

        <label for="neurological_assessments">Neurological Assessments:</label>
        <select name="neurological_assessments" required>
            <option value="normal">Normal</option>
            <option value="abnormal">Abnormal</option>
        </select>

        <label for="language_development">Language Development:</label>
        <select name="language_development" required>
            <option value="normal">Normal</option>
            <option value="delayed">Delayed</option>
        </select>

        <label for="motor_skills">Motor Skills:</label>
        <select name="motor_skills" required>
            <option value="normal">Normal</option>
            <option value="delayed">Delayed</option>
        </select>

        <label for="behavioral_assessment_score">Behavioral Assessment Score:</label>
        <input type="number" name="behavioral_assessment_score" min="0" required>

        <label for="screen_time">Screen Time (hours):</label>
        <input type="number" name="screen_time" min="0" required>

        <label for="head_circumference">Head Circumference (cm):</label>
        <input type="number" name="head_circumference" min="0" required>

        <button type="submit">Submit</button>
    </form>

    <div id="result"></div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(event) {
            event.preventDefault();

            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => { data[key] = value; });

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = 'Prediction: ' + data.prediction;
            })
            .catch(error => {
                document.getElementById('result').innerText = 'Error: ' + error.message;
            });
        });
    </script>
</body>
</html>
"""

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input from the request
    input_data = request.get_json()
    
    # Convert input data into DataFrame
    df = pd.DataFrame([input_data])

    # Apply label encoding for categorical features
    for feature in categorical_features:
        if feature in df.columns:
            df[feature] = le_dict[feature].fit_transform(df[feature].astype(str))

    # Ensure numerical columns are correctly cast
    df = df.apply(pd.to_numeric, errors='coerce')

    # Scale and apply polynomial features
    scaled_input = scaler.transform(df)
    poly_input = poly.transform(scaled_input)

    # Make prediction
    prediction = best_model.predict(poly_input)

    # Send the prediction result as a JSON response
    return jsonify({'prediction': 'Positive' if prediction[0] == 1 else 'Negative'})

# Render the form
@app.route('/')
def form():
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(debug=True)
