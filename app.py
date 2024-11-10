import numpy as np
from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model using joblib
model = joblib.load('random_forest_regressor.joblib')

# Load any necessary preprocessing objects (like encoders) if you have them
# Example: label_encoder = joblib.load('label_encoder.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    invoice_date = request.form['invoice_date']
    job_card_date = request.form['job_card_date']
    business_partner_name = request.form['business_partner_name']
    vehicle_no = request.form['vehicle_no']
    vehicle_model = request.form['vehicle_model']
    current_km_reading = float(request.form['current_km_reading'])
    invoice_line_text = request.form['invoice_line_text']

    # Create a DataFrame for the model input
    input_data = pd.DataFrame({
        'invoice_date': [invoice_date],
        'job_card_date': [job_card_date],
        'business_partner_name': [business_partner_name],
        'vehicle_no': [vehicle_no],
        'vehicle_model': [vehicle_model],
        'current_km_reading': [current_km_reading],
        'invoice_line_text': [invoice_line_text]
    })

    # Preprocess the input data
    # Example: input_data['business_partner_name'] = label_encoder.transform(input_data['business_partner_name'])
    # Make sure to apply the same preprocessing steps you used when training the model

    # For demonstration, we'll assume you have already handled the preprocessing
    # Convert categorical variables to numerical if necessary
    # This step will depend on how you encoded your categorical features during training

    # Example: Let's say you use one-hot encoding for categorical variables
    input_data = pd.get_dummies(input_data)

    # Align the columns to match the model's training data
    # Example: If your model was trained with specific columns, ensure they are in the same order
    # model_columns = [...]  # List of columns your model expects
    # input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Make the prediction
    prediction = model.predict(input_data)
    output = prediction[0]

    return render_template('index.html', prediction_text=f'Predicted Value: {output}')

if __name__ == "__main__":
    app.run(debug=True)