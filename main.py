# main.py
from app import app  # Import the app from app.py

if __name__ == '__main__':
    app.run(
        debug=True)  # Run the app with debug enabled (only for development)
