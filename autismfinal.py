from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

app = Flask(__name__)

# Generate a balanced placeholder dataset with more variety in features
X, y = make_classification(
    n_samples=300,
    n_features=6,         # Increased feature count for more variety
    n_informative=5,
    n_redundant=1,
    n_classes=2,
    weights=[0.5, 0.5],  # Ensuring balanced classes
    random_state=42
)

# Preprocessing pipeline
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2)
X_scaled = scaler.fit_transform(X)
X_poly = poly.fit_transform(X_scaled)

# Train the logistic regression model with class balancing
best_model = LogisticRegression(class_weight='balanced')
best_model.fit(X_poly, y)

# Initialize label encoders for categorical fields
le_gender = LabelEncoder().fit(['Male', 'Female'])
le_family_history = LabelEncoder().fit(['Yes', 'No'])
le_developmental_milestones = LabelEncoder().fit(['Normal', 'Delayed'])
le_social_interactions = LabelEncoder().fit(['Good', 'Poor'])
le_communication_skills = LabelEncoder().fit(['Good', 'Poor'])
le_language_development = LabelEncoder().fit(['Normal', 'Delayed'])
le_motor_skills = LabelEncoder().fit(['Normal', 'Delayed'])

# Function to preprocess user input
def preprocess_user_input(input_data):
    df = pd.DataFrame(input_data)
    # Encode categorical variables
    df['gender'] = le_gender.transform(df['gender'])
    df['family_history_asd'] = le_family_history.transform(df['family_history_asd'])
    df['developmental_milestones'] = le_developmental_milestones.transform(df['developmental_milestones'])
    df['social_interactions'] = le_social_interactions.transform(df['social_interactions'])
    df['communication_skills'] = le_communication_skills.transform(df['communication_skills'])
    df['language_development'] = le_language_development.transform(df['language_development'])
    df['motor_skills'] = le_motor_skills.transform(df['motor_skills'])
    return df

# HTML template for form (unchanged)
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
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            padding: 0;
        }
        h1 {
            text-align: center;
            color: black;
        }
        form {
            display: flex;
            flex-wrap: wrap;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            max-width: 600px;
        }
        .form-item {
            width: 50%;
            padding: 10px;
            box-sizing: border-box;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            color: #555;
        }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 14px;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #0056b3;
        }
        #result-box {
            display: none;
            max-width: 600px;
            margin-top: 20px;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            font-size: 16px;
            color: #333;
            font-family: 'Roboto', sans-serif;
            animation: fadeInUp 0.8s ease-in-out;
            opacity: 0;
            transition: opacity 0.8s ease-in-out;
        }
        #result-box.show {
            display: block;
            opacity: 1;
        }
        #result {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            color: #333;
            margin-bottom: 15px;
        }
        #additional-info {
            color: #555;
            line-height: 1.6;
        }
        .positive-result {
            background-color: #ffdddd;
            border-left: 6px solid #ff4d4d;
        }
        .negative-result {
            background-color: #ddffdd;
            border-left: 6px solid #4CAF50;
        }
        ul {
            padding-left: 20px;
            list-style-type: disc;
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
</head>
<body>
    <div>
        <h1>Autism Prediction</h1>
        <form id="predictionForm">
            {% for item in form_items %}
            <div class="form-item">
                <label for="{{ item['name'] }}">{{ item['label'] }}</label>
                {% if item['type'] == 'select' %}
                    <select name="{{ item['name'] }}" required>
                        <option value="" disabled selected>Select</option>
                        {% for option in item['options'] %}
                            <option value="{{ option }}">{{ option }}</option>
                        {% endfor %}
                    </select>
                {% else %}
                    <input type="{{ item['type'] }}" name="{{ item['name'] }}" min="0" max="100" required>
                {% endif %}
            </div>
            {% endfor %}
        </form>
        <button type="submit" form="predictionForm">Submit</button>
        <div id="result-box">
            <div id="result"></div>
            <div id="additional-info"></div>
        </div>
    </div>

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
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                const resultBox = document.getElementById('result-box');
                const resultDiv = document.getElementById('result');
                const additionalInfoDiv = document.getElementById('additional-info');

                if (data.prediction) {
                    resultDiv.innerText = "Prediction: Positive";
                    resultBox.classList.add('positive-result');
                    additionalInfoDiv.innerHTML = `
                        We are sorry to inform you that your child may be at risk for autism. We recommend the following actions to support your child:
                        <ul>
                            <li><b>Consult a Specialist:</b> Arrange a professional evaluation with a pediatrician or child psychologist to confirm the results and discuss tailored support options.</li>
                            <li><b>Monitor Developmental Progress:</b> Track your child’s social, communication, and motor skills regularly to gain insights for future interventions.</li>
                            <li><b>Explore Early Intervention Services:</b> Consider early intervention therapies, like speech or occupational therapy, to help support developmental growth and skill-building.</li>
                            <li><b>Create a Supportive Environment:</b> Foster a calm, predictable routine with positive reinforcement, which can be comforting and help reduce stress.</li>
                            <li><b>Educate Family Members:</b> Ensure that family members are aware and supportive, creating a unified approach that helps your child feel secure and understood.</li>
                            <li><b>Stay Connected and Informed:</b> Join autism support groups, both local and online, for valuable information, community support, and the latest updates in ASD research.</li>
                            <li><b>Promote Healthy Habits:</b> Ensure a balanced diet, sufficient sleep, and physical activity, which all contribute to overall well-being and support developmental progress.</li>
                        </ul>
                        These steps provide an initial roadmap to help support your child’s development while awaiting further assessment.
                    `;
                } else {
                    resultDiv.innerText = "Prediction: Negative";
                    resultBox.classList.add('negative-result');
                    additionalInfoDiv.innerText = "Congratulations! Your child shows a low risk for Autism Spectrum Disorder (ASD). Keep nurturing their growth with enriching activities, supportive routines, and joyful moments of learning and play!";
                }

                resultBox.classList.add('show');
            })
            .catch(error => {
                const resultBox = document.getElementById('result-box');
                resultBox.innerHTML = `<div id="result">Error: ${error.message}</div>`;
                resultBox.classList.add('show');
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    # Define form inputs
    form_items = [
        {"name": "age", "label": "Age:", "type": "number"},
        {"name": "gender", "label": "Gender:", "type": "select", "options": ["Male", "Female"]},
        {"name": "family_history_asd", "label": "Family history of ASD:", "type": "select", "options": ["Yes", "No"]},
        {"name": "developmental_milestones", "label": "Developmental milestones:", "type": "select", "options": ["Normal", "Delayed"]},
        {"name": "aq_score", "label": "AQ Score:", "type": "number"},
        {"name": "social_interactions", "label": "Social interactions:", "type": "select", "options": ["Good", "Poor"]},
        {"name": "communication_skills", "label": "Communication skills:", "type": "select", "options": ["Good", "Poor"]},
        {"name": "behavioral_assessment_score", "label": "Behavioral assessment score:", "type": "number"},
        {"name": "language_development", "label": "Language development:", "type": "select", "options": ["Normal", "Delayed"]},
        {"name": "motor_skills", "label": "Motor skills:", "type": "select", "options": ["Normal", "Delayed"]},
    ]
    return render_template_string(html_template, form_items=form_items)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        behavioral_score = float(data['behavioral_assessment_score'])
        if behavioral_score < 49:prediction = 0  
        elif behavioral_score >= 52:prediction = 1 
        else:
            input_data = preprocess_user_input([data])
            input_scaled = scaler.transform(input_data)
            input_poly = poly.transform(input_scaled)
            prediction = int(best_model.predict(input_poly)[0])

        # Calculate probabilities
        if prediction == 0:
            probabilities = [0.9, 0.1]  # Example probabilities for negative prediction
        else:
            probabilities = [0.1, 0.9]  # Example probabilities for positive prediction # Calculate probabilities
        if prediction == 0:
            probabilities = [0.9, 0.1]  # Example probabilities for negative prediction
        else:
            probabilities = [0.1, 0.9]  # Example probabilities for positive prediction

        return jsonify({'prediction': prediction, 'probabilities': probabilities})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
