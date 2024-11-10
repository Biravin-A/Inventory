import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model using joblib
model = joblib.load('random_forest_regressor.joblib')

# Initialize the LabelEncoder
encoder = LabelEncoder()

# Load the encoder state if you saved it during training
# If you saved the encoders for the categorical variables, load them here
# Example: encoder_bp = joblib.load('business_partner_encoder.joblib')
# Similarly for vehicle_no, vehicle_model, and invoice_line_text

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

    # Preprocess the input data
    # Convert dates to ordinal (numeric)
    invoice_date_ordinal = pd.to_datetime(invoice_date).toordinal()
    job_card_date_ordinal = pd.to_datetime(job_card_date).toordinal()

    # Encode categorical variables
    business_partner_name_encoded = encoder.fit_transform([business_partner_name])[0]
    vehicle_no_encoded = encoder.fit_transform([vehicle_no])[0]
    vehicle_model_encoded = encoder.fit_transform([vehicle_model])[0]
    invoice_line_text_encoded = encoder.fit_transform([invoice_line_text])[0]

    # Create a DataFrame for the model input
    input_data = pd.DataFrame({
        'invoice_date': [invoice_date_ordinal],
        'job_card_date': [job_card_date_ordinal],
        'business_partner_name': [business_partner_name_encoded],
        'vehicle_no': [vehicle_no_encoded],
        'vehicle_model': [vehicle_model_encoded],
        'current_km_reading': [current_km_reading],
        'invoice_line_text': [invoice_line_text_encoded]
    })

    # Make the prediction
    prediction = model.predict(input_data)
    output = prediction[0]

    return render_template('index.html', prediction_text=f'Predicted Value: {output}')

if __name__ == "__main__":
    app.run(debug=True)
