from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import sqlite3
import pickle
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = 'your_secret_key' # Required for sessions
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'artifacts' / 'model.pkl'
PREPROCESSOR_PATH = BASE_DIR / 'artifacts' / 'preprocessor.pkl'
model = None
preprocessor = None

FEATURE_COLUMNS = [
    'Age',
    'Protein1',
    'Protein2',
    'Protein3',
    'Protein4',
    'Gender',
    'Tumour_Stage',
    'Histology',
    'HER2 status',
]


def load_prediction_artifacts():
    global model, preprocessor

    if model is None:
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)

    if preprocessor is None:
        with open(PREPROCESSOR_PATH, 'rb') as preprocessor_file:
            preprocessor = pickle.load(preprocessor_file)

    return model, preprocessor


def get_alive_probability(model_obj, transformed_data, prediction):
    if hasattr(model_obj, 'predict_proba'):
        probabilities = model_obj.predict_proba(transformed_data)[0]
        classes = list(model_obj.classes_)

        if 1 in classes:
            return float(probabilities[classes.index(1)])
        if 1.0 in classes:
            return float(probabilities[classes.index(1.0)])

    return 1.0 if int(prediction) == 1 else 0.0

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users 
                      (id INTEGER PRIMARY KEY, email TEXT, username TEXT, password TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        hashed_pw = generate_password_hash(data['password'])
        
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)",
                           (data['email'], data['username'], hashed_pw))
            conn.commit()
            return jsonify({"status": "success"})
        except:
            return jsonify({"status": "error", "message": "User already exists"})
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (data['username'],))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], data['password']):
            session['user'] = user[2]
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Invalid Credentials"})
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['user'])

@app.route('/analytics')
def analytics():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('analytics.html', username=session['user'])

@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('about.html', username=session['user'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        patient_data = {
            'Age': float(data.get('Age', 0)),
            'Protein1': float(data.get('Protein1', 0)),
            'Protein2': float(data.get('Protein2', 0)),
            'Protein3': float(data.get('Protein3', 0)),
            'Protein4': float(data.get('Protein4', 0)),
            'Gender': data.get('Gender', 'FEMALE'),
            'Tumour_Stage': data.get('Tumour_Stage', 'II'),
            'Histology': data.get('Histology', 'Infiltrating Ductal Carcinoma'),
            'HER2 status': data.get('HER2 status', 'Negative'),
        }

        if patient_data['Age'] <= 0 or patient_data['Age'] > 120:
            return jsonify({'status': 'error', 'message': 'Age must be between 1 and 120.'}), 400

        input_df = pd.DataFrame([patient_data], columns=FEATURE_COLUMNS)
        model_obj, preprocessor_obj = load_prediction_artifacts()
        transformed_data = preprocessor_obj.transform(input_df)
        prediction = model_obj.predict(transformed_data)[0]
        alive_probability = get_alive_probability(model_obj, transformed_data, prediction)
        survival_percent = round(alive_probability * 100, 1)
        status = 'Alive' if int(prediction) == 1 else 'Dead'
        risk_score = 100 - survival_percent

        if survival_percent >= 70:
            risk_level = 'Low Risk'
        elif survival_percent >= 40:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'High Risk'

        return jsonify({
            'status': 'success',
            'prediction': status,
            'survival_probability': survival_percent,
            'risk_score': round(risk_score, 1),
            'risk_level': risk_level,
            'patient': patient_data,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)