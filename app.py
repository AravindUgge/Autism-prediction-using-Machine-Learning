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

# HTML template for form with world-class UI
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="World's leading AI-powered Autism Prediction Tool">
    <meta name="keywords" content="autism, prediction, AI, health, screening">
    <meta name="author" content="xAI">
    <title>Autism Prediction - Powered by xAI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #1e3c72, #2a5298, #6e8efb);
            min-height: 100vh;
            color: #333;
            transition: background 0.5s ease;
        }
        body.dark-theme {
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            color: #e0e0e0;
        }
        .header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 2000;
        }
        .dark-theme .header {
            background: rgba(26, 26, 46, 0.95);
        }
        .logo {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2a5298;
        }
        .dark-theme .logo {
            color: #6e8efb;
        }
        .nav a {
            color: #1e3c72;
            text-decoration: none;
            margin: 0 15px;
            font-size: 1rem;
            font-weight: 500;
            transition: color 0.3s ease;
        }
        .dark-theme .nav a {
            color: #e0e0e0;
        }
        .nav a:hover {
            color: #6e8efb;
        }
        .theme-toggle {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: #1e3c72;
        }
        .dark-theme .theme-toggle {
            color: #e0e0e0;
        }
        .hero {
            text-align: center;
            padding: 100px 20px 60px;
            background: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)), url('https://source.unsplash.com/1600x900/?health');
            background-size: cover;
            color: #fff;
            animation: fadeIn 1.5s ease-out;
        }
        .hero h1 {
            font-size: 3rem;
            margin-bottom: 20px;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        .hero p {
            font-size: 1.2rem;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            padding: 40px;
            animation: slideUp 1s ease-out;
        }
        .dark-theme .container {
            background: rgba(26, 26, 46, 0.98);
            color: #e0e0e0;
        }
        .progress-container {
            width: 100%;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            margin-bottom: 30px;
            overflow: hidden;
        }
        .dark-theme .progress-container {
            background: #3a3a5c;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #2a5298, #6e8efb);
            width: 0;
            transition: width 0.4s ease;
        }
        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 25px;
        }
        .form-item {
            position: relative;
        }
        label {
            font-size: 0.95rem;
            font-weight: 500;
            color: #1e3c72;
            margin-bottom: 10px;
            display: block;
            transition: color 0.3s ease;
        }
        .dark-theme label {
            color: #e0e0e0;
        }
        input, select {
            width: 100%;
            padding: 14px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        .dark-theme input, .dark-theme select {
            background: #3a3a5c;
            border-color: #4a4a7c;
            color: #e0e0e0;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #6e8efb;
            box-shadow: 0 0 15px rgba(110, 142, 251, 0.3);
            background: #fff;
        }
        .dark-theme input:focus, .dark-theme select:focus {
            background: #4a4a7c;
        }
        .form-item::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 0;
            height: 3px;
            background: #6e8efb;
            transition: width 0.4s ease;
        }
        .form-item:focus-within::after {
            width: 100%;
        }
        .tooltip {
            position: absolute;
            top: -40px;
            left: 50%;
            transform: translateX(-50%);
            background: #2a5298;
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        }
        .form-item:hover .tooltip {
            opacity: 1;
            visibility: visible;
            top: -50px;
        }
        button {
            grid-column: span 2;
            padding: 18px;
            background: linear-gradient(90deg, #2a5298, #6e8efb);
            color: #fff;
            border: none;
            border-radius: 50px;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .dark-theme button {
            background: linear-gradient(90deg, #6e8efb, #a777e3);
        }
        button:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(110, 142, 251, 0.4);
        }
        button:disabled {
            background: #b0b0b0;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .loader {
            display: none;
            border: 6px solid #f3f3f3;
            border-top: 6px solid #2a5298;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        .dark-theme .loader {
            border-color: #3a3a5c;
            border-top-color: #6e8efb;
        }
        .loader.show {
            display: block;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            justify-content: center;
            align-items: center;
            z-index: 3000;
        }
        .modal-content {
            background: #fff;
            border-radius: 20px;
            padding: 40px;
            max-width: 700px;
            width: 90%;
            animation: slideUp 0.6s ease-out;
            position: relative;
        }
        .dark-theme .modal-content {
            background: #1a1a2e;
            color: #e0e0e0;
        }
        .close-btn {
            position: absolute;
            top: 15px;
            right: 20px;
            font-size: 2rem;
            color: #1e3c72;
            cursor: pointer;
            transition: color 0.3s ease;
        }
        .dark-theme .close-btn {
            color: #e0e0e0;
        }
        .close-btn:hover {
            color: #ff4d4d;
        }
        #result {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            color: #1e3c72;
            margin-bottom: 20px;
        }
        .dark-theme #result {
            color: #e0e0e0;
        }
        #additional-info {
            color: #34495e;
            line-height: 1.8;
            font-size: 1.1rem;
        }
        .dark-theme #additional-info {
            color: #b0b0d0;
        }
        .confidence-meter {
            margin: 20px 0;
            text-align: center;
        }
        .meter {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        .dark-theme .meter {
            background: #3a3a5c;
        }
        .meter-fill {
            height: 100%;
            background: linear-gradient(90deg, #2a5298, #6e8efb);
            transition: width 0.5s ease;
        }
        .positive-result {
            background: linear-gradient(135deg, #ffdddd, #ffe6e6);
            border-left: 10px solid #ff4d4d;
        }
        .dark-theme .positive-result {
            background: linear-gradient(135deg, #4a1a1a, #5a2a2a);
        }
        .negative-result {
            background: linear-gradient(135deg, #ddffdd, #e6ffe6);
            border-left: 10px solid #4CAF50;
        }
        .dark-theme .negative-result {
            background: linear-gradient(135deg, #1a4a1a, #2a5a2a);
        }
        ul {
            padding-left: 25px;
            list-style-type: disc;
        }
        footer {
            text-align: center;
            padding: 40px 20px;
            background: #1e3c72;
            color: #fff;
            margin-top: 40px;
        }
        .dark-theme footer {
            background: #0f3460;
        }
        footer a {
            color: #6e8efb;
            text-decoration: none;
            margin: 0 10px;
        }
        footer a:hover {
            text-decoration: underline;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(60px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @media (max-width: 768px) {
            form {
                grid-template-columns: 1fr;
            }
            button {
                grid-column: span 1;
            }
            .hero h1 {
                font-size: 2rem;
            }
            .container {
                margin: 20px;
                padding: 20px;
            }
        }
        [aria-hidden="true"] {
            display: none;
        }
        [aria-hidden="false"] {
            display: flex;
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
</head>
<body>
    <header class="header" role="banner">
        <div class="logo">Autism Predictor</div>
        <nav class="nav" role="navigation">
            <a href="#home">Home</a>
            <a href="#about">About</a>
            <a href="#resources">Resources</a>
            <button class="theme-toggle" aria-label="Toggle dark theme">🌙</button>
        </nav>
    </header>
    <section class="hero" id="home">
        <h1>AI-Powered Autism Screening</h1>
        <p>Discover insights into autism risk with our state-of-the-art machine learning tool, designed to provide accurate and compassionate predictions.</p>
    </section>
    <main class="container" role="main">
        <div class="progress-container">
            <div class="progress-bar" id="progress-bar"></div>
        </div>
        <form id="predictionForm" aria-label="Autism Prediction Form">
            {% for item in form_items %}
            <div class="form-item">
                <label for="{{ item['name'] }}">{{ item['label'] }}</label>
                {% if item['type'] == 'select' %}
                    <select name="{{ item['name'] }}" id="{{ item['name'] }}" required aria-required="true">
                        <option value="" disabled selected>Select an option</option>
                        {% for option in item['options'] %}
                            <option value="{{ option }}">{{ option }}</option>
                        {% endfor %}
                    </select>
                    <span class="tooltip">Select {{ item['label'] | replace(':', '') | lower }}</span>
                {% else %}
                    <input type="{{ item['type'] }}" name="{{ item['name'] }}" id="{{ item['name'] }}" min="0" max="100" required aria-required="true">
                    <span class="tooltip">Enter {{ item['label'] | replace(':', '') | lower }}</span>
                {% endif %}
            </div>
            {% endfor %}
        </form>
        <button type="submit" form="predictionForm" id="submit-btn" aria-label="Submit Prediction Form">Get Prediction</button>
        <div class="loader" id="loader" aria-hidden="true"></div>
        <div class="modal" id="result-modal" aria-hidden="true">
            <div class="modal-content">
                <span class="close-btn" aria-label="Close modal">×</span>
                <div id="result"></div>
                <div class="confidence-meter">
                    <p>Confidence Score</p>
                    <div class="meter">
                        <div class="meter-fill" id="confidence-fill"></div>
                    </div>
                </div>
                <div id="additional-info"></div>
            </div>
        </div>
    </main>
    <footer id="resources">
        <p>Resources: <a href="https://www.autismspeaks.org">Autism Speaks</a> | <a href="https://www.cdc.gov/ncbddd/autism">CDC Autism</a> | <a href="https://www.autism.org">Autism Society</a></p>
        <p>&copy; 2025 Aravind Ugge. All rights reserved.</p>
    </footer>

    <script>
        // Theme toggle
        const themeToggle = document.querySelector('.theme-toggle');
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-theme');
            themeToggle.textContent = document.body.classList.contains('dark-theme') ? '☀️' : '🌙';
        });

        // Progress bar update
        const inputs = document.querySelectorAll('#predictionForm input, #predictionForm select');
        const progressBar = document.getElementById('progress-bar');
        const totalInputs = inputs.length;

        function updateProgress() {
            const filledInputs = Array.from(inputs).filter(input => input.value !== '').length;
            const progress = (filledInputs / totalInputs) * 100;
            progressBar.style.width = `${progress}%`;
        }

        inputs.forEach(input => {
            input.addEventListener('input', updateProgress);
        });

        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', function(event) {
            event.preventDefault();
            const loader = document.getElementById('loader');
            const modal = document.getElementById('result-modal');
            const submitBtn = document.getElementById('submit-btn');
            loader.setAttribute('aria-hidden', 'false');
            modal.setAttribute('aria-hidden', 'true');
            submitBtn.disabled = true;

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
                loader.setAttribute('aria-hidden', 'true');
                submitBtn.disabled = false;
                const resultDiv = document.getElementById('result');
                const additionalInfoDiv = document.getElementById('additional-info');
                const modalContent = document.querySelector('.modal-content');
                const confidenceFill = document.getElementById('confidence-fill');
                const confidence = data.probabilities[data.prediction] * 100;

                confidenceFill.style.width = `${confidence}%`;

                if (data.prediction) {
                    resultDiv.innerText = "Prediction: Positive";
                    modalContent.classList.add('positive-result');
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
                    modalContent.classList.add('negative-result');
                    additionalInfoDiv.innerText = "Congratulations! Your child shows a low risk for Autism Spectrum Disorder (ASD). Keep nurturing their growth with enriching activities, supportive routines, and joyful moments of learning and play!";
                }

                modal.setAttribute('aria-hidden', 'false');
            })
            .catch(error => {
                loader.setAttribute('aria-hidden', 'true');
                submitBtn.disabled = false;
                const resultDiv = document.getElementById('result');
                resultDiv.innerText = `Error: ${error.message}`;
                modal.setAttribute('aria-hidden', 'false');
            });
        });

        // Modal close functionality
        document.querySelector('.close-btn').addEventListener('click', () => {
            const modal = document.getElementById('result-modal');
            modal.setAttribute('aria-hidden', 'true');
            document.querySelector('.modal-content').classList.remove('positive-result', 'negative-result');
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
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
