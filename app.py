from flask import Flask, request, render_template, flash
import numpy as np
import pickle


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Load the model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = [x for x in request.form.values()]

    # Check for missing values
    if '' in input_data:
        flash('Please fill out all fields before submitting.', 'error')
        return render_template('predictor.html')

    # Convert input data to float
    input_data = [float(x) for x in input_data]

    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    # Make prediction and get probability
    prediction = model.predict(std_data)
    probability = model.predict_proba(std_data)[0][1] * 100  # Probability of being diabetic

    # Determine the result and provide suggestions based on the risk percentage
    if probability >= 75:
        result = f"The person is diabetic with a very high risk percentage of {probability:.2f}%."
        suggestions = (
            "It is highly recommended to consult with a healthcare provider immediately. "
            "Consider starting medication as prescribed by your doctor, along with a strict diet plan. "
            "Focus on reducing high sugar and high carbohydrate foods. Regular physical activity, such as walking or swimming, is crucial. "
            "Monitoring blood sugar levels daily and managing stress through mindfulness techniques are also recommended."
        )
    elif 50 <= probability < 75:
        result = f"The person is diabetic with a moderate risk percentage of {probability:.2f}%."
        suggestions = (
            "You should consult with a healthcare provider to discuss preventive measures. "
            "A balanced diet with controlled portions of carbohydrates and sugars is important. "
            "Include at least 30 minutes of moderate physical activity like brisk walking in your daily routine. "
            "Monitor your blood sugar levels regularly and manage stress through relaxation techniques."
        )
    elif 25 <= probability < 50:
        result = f"The person is at a low to moderate risk of diabetes with a risk percentage of {probability:.2f}%."
        suggestions = (
            "Maintaining a healthy lifestyle can help reduce your risk. Focus on a diet rich in fruits, vegetables, lean proteins, and whole grains. "
            "Regular physical activity, such as walking or yoga, can help maintain healthy blood sugar levels. "
            "Consider regular check-ups with your healthcare provider and monitor your weight and blood sugar levels periodically."
        )
    else:
        result = f"The person is not diabetic with a low risk percentage of {probability:.2f}%."
        suggestions = (
            "Continue maintaining a healthy lifestyle with a balanced diet and regular exercise. "
            "Even with a low risk, it's important to have regular check-ups to monitor your overall health. "
            "Avoid high sugar and processed foods, and include plenty of fiber-rich foods in your diet. "
            "Stay active with at least 30 minutes of physical activity daily."
        )

    return render_template('predictor.html', prediction_text=result, suggestions_text=suggestions)

if __name__ == "__main__":
    app.run(debug=True,port= 5001)

