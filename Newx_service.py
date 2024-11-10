import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('random_forest_regressor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('index.html', prediction_text=f'Predicted Value: {output}')

if __name__ == "__main__":
    app.run(debug=True)
