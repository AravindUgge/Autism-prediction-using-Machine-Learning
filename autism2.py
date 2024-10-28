from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

app = Flask(_name_)

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
    <title>ASD Prediction</title>
</head>
<body>
    <h1>ASD Prediction Form</h1>
    <form id="predictionForm">
        <label for="age">Age:</label>
        <input type="number" name="age" required><br>

        <label for="gender">Gender:</label>
        <input type="text" name="gender" required><br>

        <label for="ethnicity">Ethnicity:</label>
        <input type="text" name="ethnicity" required><br>

        <label for="family_history_asd">Family history of ASD (yes/no):</label>
        <input type="text" name="family_history_asd" required><br>

        <label for="pre_existing_conditions">Pre-existing conditions:</label>
        <input type="text" name="pre_existing_conditions" required><br>

        <label for="developmental_milestones">Developmental milestones:</label>
        <input type="text" name="developmental_milestones" required><br>

        <label for="aq_score">AQ Score:</label>
        <input type="number" name="aq_score" required><br>

        <label for="social_interactions">Social interactions:</label>
        <input type="text" name="social_interactions" required><br>

        <label for="communication_skills">Communication skills:</label>
        <input type="text" name="communication_skills" required><br>

        <label for="genetic_markers">Genetic markers:</label>
        <input type="text" name="genetic_markers" required><br>

        <label for="genomic_test_results">Genomic test results:</label>
        <input type="text" name="genomic_test_results" required><br>

        <label for="environmental_toxins">Environmental toxins (yes/no):</label>
        <input type="text" name="environmental_toxins" required><br>

        <label for="prenatal_factors">Prenatal factors:</label>
        <input type="text" name="prenatal_factors" required><br>

        <label for="psychological_evaluations">Psychological evaluations:</label>
        <input type="text" name="psychological_evaluations" required><br>

        <label for="neurological_assessments">Neurological assessments:</label>
        <input type="text" name="neurological_assessments" required><br>

        <label for="language_development">Language development:</label>
        <input type="text" name="language_development" required><br>

        <label for="motor_skills">Motor skills:</label>
        <input type="text" name="motor_skills" required><br>

        <label for="behavioral_assessment_score">Behavioral assessment score:</label>
        <input type="number" name="behavioral_assessment_score" required><br>

        <label for="screen_time">Daily screen time:</label>
        <input type="number" name="screen_time" required><br>

        <label for="parenting_style">Parenting style:</label>
        <input type="text" name="parenting_style" required><br>

        <label for="sleep_quality">Sleep quality:</label>
        <input type="text" name="sleep_quality" required><br>

        <label for="dietary_habits">Dietary habits:</label>
        <input type="text" name="dietary_habits" required><br>

        <label for="physical_activity">Physical activity level:</label>
        <input type="text" name="physical_activity" required><br>

        <label for="social_engagement">Social engagement:</label>
        <input type="text" name="social_engagement" required><br>

        <label for="emotional_regulation">Emotional regulation:</label>
        <input type="text" name="emotional_regulation" required><br>

        <label for="technology_use">Technology use:</label>
        <input type="text" name="technology_use" required><br>

        <button type="submit">Submit</button>
    </form>

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
    # Collect user input data from JSON request
    user_input = {
        'age': [float(request.json['age'])],
        'gender': [request.json['gender']],
        'ethnicity': [request.json['ethnicity']],
        'family_history_asd': [request.json['family_history_asd']],
        'pre_existing_conditions': [request.json['pre_existing_conditions']],
        'developmental_milestones': [request.json['developmental_milestones']],
        'aq_score': [int(request.json['aq_score'])],
        'social_interactions': [request.json['social_interactions']],
        'communication_skills': [request.json['communication_skills']],
        'genetic_markers': [request.json['genetic_markers']],
        'genomic_test_results': [request.json['genomic_test_results']],
        'environmental_toxins': [request.json['environmental_toxins']],
        'prenatal_factors': [request.json['prenatal_factors']],
        'psychological_evaluations': [request.json['psychological_evaluations']],
        'neurological_assessments': [request.json['neurological_assessments']],
        'language_development': [request.json['language_development']],
        'motor_skills': [request.json['motor_skills']],
        'behavioral_assessment_score': [int(request.json['behavioral_assessment_score'])],
        'screen_time': [int(request.json['screen_time'])],
        'parenting_style': [request.json['parenting_style']],
        'sleep_quality': [request.json['sleep_quality']],
        'dietary_habits': [request.json['dietary_habits']],
        'physical_activity': [request.json['physical_activity']],
        'social_engagement': [request.json['social_engagement']],
        'emotional_regulation': [request.json['emotional_regulation']],
        'technology_use': [request.json['technology_use']],
    }

    # Preprocess user input
    user_data = preprocess_user_input(pd.DataFrame(user_input))

    # Make a prediction
    prediction = best_model.predict(poly.transform(scaler.transform(user_data)))[0]

    return jsonify({'prediction': prediction})

if _name_ == '_main_':
    app.run(debug=True)
