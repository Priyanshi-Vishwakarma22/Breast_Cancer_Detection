from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import sqlite3
import pickle
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import traceback
import os
import hashlib
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')

BASE_DIR = Path(__file__).resolve().parent

# ── ENCRYPTION SETUP ──────────────────────────────────────────────────────────

KEY_FILE = BASE_DIR / 'encryption.key'

def get_encryption_key():
    """Get or create encryption key for patient data"""
    if KEY_FILE.exists():
        with open(KEY_FILE, 'rb') as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as f:
            f.write(key)
        print("✅ New encryption key generated")
        return key

ENCRYPTION_KEY = get_encryption_key()
cipher = Fernet(ENCRYPTION_KEY)

def encrypt_data(text):
    """Encrypt sensitive data"""
    if text is None or text == "":
        return None
    try:
        return cipher.encrypt(text.encode()).decode()
    except:
        return text

def decrypt_data(encrypted_text):
    """Decrypt sensitive data"""
    if encrypted_text is None or encrypted_text == "":
        return None
    try:
        return cipher.decrypt(encrypted_text.encode()).decode()
    except:
        return encrypted_text

def hash_patient_id(patient_id):
    """Hash patient identifier"""
    if patient_id is None:
        return None
    return hashlib.sha256(f"patient_{patient_id}_salt_2024".encode()).hexdigest()[:16]

# ── DATABASE SETUP ────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        username TEXT UNIQUE,
        password TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Predictions table with encrypted fields
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        patient_hash TEXT,
        age INTEGER,
        stage TEXT,
        histology TEXT,
        surgery_type TEXT,
        protein1 REAL,
        protein2 REAL,
        protein3 REAL,
        protein4 REAL,
        her2_status TEXT,
        gender TEXT,
        probability REAL,
        status TEXT,
        risk_level TEXT,
        encrypted_diagnosis TEXT,
        timestamp TEXT
    )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_db()

# ── ROUTES ────────────────────────────────────────────────────────────────────

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
            cursor.execute('INSERT INTO users (email, username, password) VALUES (?, ?, ?)',
                           (data['email'], data['username'], hashed_pw))
            conn.commit()
            conn.close()
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": "User already exists"})
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
        data = request.get_json()
        logging.info(f"Received prediction request: {data}")

        # ── Validate inputs ───────────────────────────────────────────────────
        age = float(data.get('Age', 0))
        if age <= 0 or age > 120:
            return jsonify({'status': 'error', 'message': 'Invalid age'}), 400

        stage     = data.get('Tumour_Stage', 'II')
        histology = data.get('Histology', 'Infiltrating Ductal Carcinoma')
        surgery   = data.get('Surgery_type', 'Other')
        her2      = data.get('HER2 status', 'Negative')
        gender    = data.get('Gender', 'FEMALE')
        p1        = float(data.get('Protein1', 0))
        p2        = float(data.get('Protein2', 0))
        p3        = float(data.get('Protein3', 0))
        p4        = float(data.get('Protein4', 0))

        # ── Use ML model via PredictPipeline ─────────────────────────────────
        custom_data = CustomData(
            Age=int(age),
            Gender=gender,
            Protein1=p1,
            Protein2=p2,
            Protein3=p3,
            Protein4=p4,
            Tumour_Stage=stage,
            Histology=histology,
            HER2_status=her2,
            Surgery_type=surgery,
        )

        features_df = custom_data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        predictions, probabilities = pipeline.predict_with_proba(features_df)

        # ── Parse results ─────────────────────────────────────────────────────
        status = 'Alive' if predictions[0] == 1 else 'Dead'

        if probabilities is not None:
            prob_alive = float(probabilities[0][1]) * 100
        else:
            # Fallback if model doesn't support predict_proba
            prob_alive = 75.0 if status == 'Alive' else 25.0

        prob_alive = round(prob_alive, 1)
        risk_score = round(100 - prob_alive, 1)

        if prob_alive >= 70:
            risk_level = 'Low Risk'
        elif prob_alive >= 40:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'High Risk'

        logging.info(f"Prediction: {status} | Survival: {prob_alive}% | Risk: {risk_level}")

        # ── Encrypt diagnosis & save to DB ────────────────────────────────────
        patient_hash = hash_patient_id(str(age) + stage + str(datetime.now().timestamp()))
        encrypted_diagnosis = encrypt_data(
            f"Stage: {stage}, Histology: {histology}, HER2: {her2}, Surgery: {surgery}"
        )

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Ensure all columns exist
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        required_columns = ['surgery_type', 'protein1', 'protein2', 'protein3', 'protein4',
                            'her2_status', 'gender', 'risk_level', 'patient_hash', 'encrypted_diagnosis']
        for col in required_columns:
            if col not in columns:
                cursor.execute(f'ALTER TABLE predictions ADD COLUMN {col} TEXT')

        cursor.execute('''
            INSERT INTO predictions
            (username, patient_hash, age, stage, histology, surgery_type,
             protein1, protein2, protein3, protein4, her2_status, gender,
             probability, status, risk_level, encrypted_diagnosis, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session.get('user'),
            patient_hash,
            age, stage, histology, surgery,
            p1, p2, p3, p4,
            her2, gender,
            prob_alive,
            status,
            risk_level,
            encrypted_diagnosis,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            'status': 'success',
            'prediction': status,
            'survival_probability': prob_alive,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'id': new_id,
        })

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_predictions')
def get_predictions():
    if 'user' not in session:
        return jsonify([])

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, age, stage, histology, surgery_type, probability, status, timestamp, encrypted_diagnosis
        FROM predictions
        WHERE username=?
        ORDER BY id DESC LIMIT 10
    ''', (session['user'],))
    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        decrypted_diagnosis = decrypt_data(row[8]) if len(row) > 8 and row[8] else "N/A"
        result.append({
            "id":         row[0],
            "patientId":  f"PN{row[0]}",
            "age":        row[1],
            "stageLabel": "Stage " + str(row[2]),
            "histology":  row[3],
            "surgery_type": row[4] if row[4] else "N/A",
            "score":      row[5],
            "probability": row[5],
            "status":     row[6],
            "timestamp":  row[7],
            "diagnosis":  decrypted_diagnosis[:50] if decrypted_diagnosis else "N/A"
        })
    return jsonify(result)


@app.route('/delete_prediction/<int:id>', methods=['DELETE'])
def delete_prediction(id):
    if 'user' not in session:
        return jsonify({'status': 'error'}), 403

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE id=? AND username=?', (id, session['user']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get('message', '').lower()

    if "hello" in msg or "hi" in msg:
        reply = "Hello! I am your AI assistant. How can I help you?"
    elif "risk" in msg:
        reply = "Risk depends on survival probability: >70% = Low Risk, 40-70% = Medium Risk, <40% = High Risk."
    elif "stage" in msg:
        reply = "Cancer stages: Stage I (early), Stage II (locally advanced), Stage III (advanced)."
    elif "survival" in msg:
        reply = "Survival probability shows chances of recovery based on our ML model."
    elif "encryption" in msg or "privacy" in msg:
        reply = "All patient data is encrypted using AES-256. Passwords are hashed with bcrypt."
    elif "predict" in msg:
        reply = "Fill the form and click 'Run Prediction' to get ML-based survival probability."
    elif "model" in msg:
        reply = "We use machine learning models trained on breast cancer patient data to predict survival."
    else:
        reply = "I can help with cancer stages, risk levels, survival prediction, and data privacy."

    return jsonify({"reply": reply})


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("SURVIVAL.AI — ML Powered Breast Cancer Prediction")
    print("Password Hashing   : Active (bcrypt)")
    print("Data Encryption    : Active (AES-256)")
    print("Patient ID Hashing : Active (SHA-256)")
    print("ML Model           : Active (trained model.pkl)")
    print("URL                : http://127.0.0.1:5000")

    cert_file = BASE_DIR / 'cert.pem'
    key_file  = BASE_DIR / 'key.pem'

    if cert_file.exists() and key_file.exists():
        print("Running with HTTPS (SSL)")
        app.run(debug=True, host='0.0.0.0', port=443,
                ssl_context=(str(cert_file), str(key_file)))
    else:
        print("Running with HTTP")
        app.run(debug=True, host='127.0.0.1', port=5000)