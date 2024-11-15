# app.py
from flask import Flask, request, render_template
from urllib.parse import quote as url_quote
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_regressor.joblib')

# Load the scaler if it was used during model training
scaler = joblib.load('scaler.joblib')  # Load the scaler

# Load any label encoders (if needed, you should have saved the encoders during training)
# label_encoder = joblib.load('label_encoder.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        invoice_date = request.form['invoice_date']
        job_card_date = request.form['job_card_date']
        business_partner_name = request.form['business_partner_name']
        vehicle_no = request.form['vehicle_no']
        vehicle_model = request.form['vehicle_model']
        current_km_reading = float(request.form['current_km_reading'])
        invoice_line_text = request.form['invoice_line_text']

        # Ensure all required fields are provided
        if not invoice_date or not job_card_date or not business_partner_name:
            raise ValueError("Missing required input fields")

        # Convert dates from strings to datetime objects
        invoice_date = datetime.strptime(invoice_date, '%Y-%m-%d')
        job_card_date = datetime.strptime(job_card_date, '%Y-%m-%d')

        # Extract useful features from the dates
        invoice_day = invoice_date.day
        invoice_month = invoice_date.month
        invoice_year = invoice_date.year
        job_card_day = job_card_date.day
        job_card_month = job_card_date.month
        job_card_year = job_card_date.year

        # Calculate date difference (e.g., days between invoice and job card date)
        days_diff = (job_card_date - invoice_date).days

        # Create DataFrame from the form data
        input_data = pd.DataFrame({
            'invoice_date': [invoice_date],
            'job_card_date': [job_card_date],
            'business_partner_name': [business_partner_name],
            'vehicle_no': [vehicle_no],
            'vehicle_model': [vehicle_model],
            'current_km_reading': [current_km_reading],
            'invoice_line_text': [invoice_line_text],
            'invoice_day': [invoice_day],
            'invoice_month': [invoice_month],
            'invoice_year': [invoice_year],
            'job_card_day': [job_card_day],
            'job_card_month': [job_card_month],
            'job_card_year': [job_card_year],
            'days_diff': [days_diff]
        })

        # Preprocessing: Handle categorical variables (e.g., One-Hot Encoding)
        input_data = pd.get_dummies(input_data, columns=['business_partner_name', 'vehicle_model'])

        # Apply scaling to 'current_km_reading' if the model was trained with it scaled
        if 'current_km_reading' in input_data.columns:
            input_data['current_km_reading'] = scaler.transform(input_data[['current_km_reading']])

        # Define the model's expected column names (based on the training)
        model_columns = [
            'invoice_day', 'invoice_month', 'invoice_year', 
            'current_km_reading', 'business_partner_name_X', 
            'vehicle_model_Y', 'days_diff'
        ]  # Replace with actual columns from training

        # Check for missing columns and fill them with zero if needed
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # Fill missing column with 0s

        # Reorder columns to match the model's training columns
        input_data = input_data[model_columns]

        # Make the prediction
        prediction = model.predict(input_data)
        output = prediction[0]

        # Render the result in the template
        return render_template('index.html', prediction_text=f'Predicted Vehicle Cost: {output}')

    except ValueError as e:
        # Handle missing input fields or conversion errors
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}. Please ensure all fields are filled out correctly."
        )

    except KeyError as e:
        # Handle missing columns in the DataFrame (in case get_dummies() doesn't generate the right columns)
        return render_template(
            'index.html',
            prediction_text=f"Error: Missing column for {str(e)}. Please check input data."
        )

    except Exception as e:
        # Catch any other exceptions
        return render_template('index.html', prediction_text=f"An error occurred: {str(e)}")

