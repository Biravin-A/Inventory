# Load the Random Forest model using pickle
with open('random_forest_regressor.pkl', 'rb') as file:
    loaded_rf_model = pickle.load(file)

print("Random Forest Regressor model loaded successfully.")