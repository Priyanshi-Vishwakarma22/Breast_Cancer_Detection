
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import sqlite3
import pickle
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime



app = Flask(__name__)
app.secret_key = 'your_secret_key'


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
    'Surgery_Type'
]

# ================= LOAD MODEL =================
def load_prediction_artifacts():
    global model, preprocessor

    if model is None:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

    if preprocessor is None:
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)

    return model, preprocessor


def get_alive_probability(model_obj, transformed_data, prediction):
    if hasattr(model_obj, 'predict_proba'):
        probs = model_obj.predict_proba(transformed_data)[0]
        classes = list(model_obj.classes_)

        if 1 in classes:
            return float(probs[classes.index(1)])

    return 1.0 if int(prediction) == 1 else 0.0


# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # USERS TABLE
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        username TEXT UNIQUE,
        password TEXT
    )
    ''')

    # PREDICTIONS TABLE
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        age INTEGER,
        stage TEXT,
        histology TEXT,
        probability REAL,
        status TEXT,
        timestamp TEXT
    )
    ''')

    conn.commit()
    conn.close()


init_db()


# ================= ROUTES =================
@app.route('/')
def home():
    return redirect(url_for('login'))


# -------- REGISTER --------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()

        hashed_pw = generate_password_hash(data['password'])

        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()

            cursor.execute('''
            INSERT INTO users (email, username, password)
            VALUES (?, ?, ?)
            ''', (data['email'], data['username'], hashed_pw))

            conn.commit()
            conn.close()

            return jsonify({"status": "success"})

        except:
            return jsonify({"status": "error", "message": "User already exists"})

    return render_template('register.html')


# -------- LOGIN --------
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


# -------- DASHBOARD --------
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template('dashboard.html', username=session['user'])


# -------- ANALYTICS --------
@app.route('/analytics')
def analytics():
    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template('analytics.html', username=session['user'])


# -------- ABOUT --------
@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template('about.html', username=session['user'])


# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        patient_data = {
            'Age': float(data.get('Age', 0)),
            'Protein1': float(data.get('Protein1', 0)),
            'Protein2': float(data.get('Protein2', 0)),
            'Protein3': float(data.get('Protein3', 0)),
            'Protein4': float(data.get('Protein4', 0)),
            'Gender': data.get('Gender', 'FEMALE'),
            'Tumour_Stage': data.get('Tumour_Stage', 'II'),
            'Histology': data.get('Histology', ''),
            'HER2 status': data.get('HER2 status', 'Negative'),
            'Surgery_Type': data.get('Surgery_Type', 'Lumpectomy')
        }

        if patient_data['Age'] <= 0 or patient_data['Age'] > 120:
            return jsonify({'status': 'error', 'message': 'Invalid age'}), 400

        df = pd.DataFrame([patient_data], columns=FEATURE_COLUMNS)

        model_obj, preprocessor_obj = load_prediction_artifacts()
        transformed = preprocessor_obj.transform(df)

        prediction = model_obj.predict(transformed)[0]
        prob = get_alive_probability(model_obj, transformed, prediction)

        survival_percent = round(prob * 100, 1)
        status = 'Alive' if int(prediction) == 1 else 'Dead'
        risk_score = 100 - survival_percent

        if survival_percent >= 70:
            risk_level = 'Low Risk'
        elif survival_percent >= 40:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'High Risk'

        # ===== SAVE TO DATABASE =====
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO predictions (username, age, stage, histology, probability, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session.get('user'),
            patient_data['Age'],
            patient_data['Tumour_Stage'],
            patient_data['Histology'],
            survival_percent,
            status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        new_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return jsonify({
            'status': 'success',
            'prediction': status,
            'survival_probability': survival_percent,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'id': new_id,
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_predictions')
def get_predictions():
    if 'user' not in session:
        return jsonify([])

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT id, age, stage, histology, probability, status, timestamp
    FROM predictions
    WHERE username=?
    ORDER BY id DESC LIMIT 8
    ''', (session['user'],))

    rows = cursor.fetchall()
    conn.close()

    data = []

    for row in rows:
        data.append({
    "id": row[0],                 # ✅ ADD THIS
    "patientId": f"PN{row[0]}",   # (already there, keep it)
    "age": row[1],
    "stageLabel": "Stage " + row[2],
    "histology": row[3],
    "surgery": row[4],
    "score": row[5],
    "status": "Alive" if row[6] == "Alive" else "Deceased",
    "timestamp": row[7]
})

    return jsonify(data)
@app.route('/delete_prediction/<int:id>', methods=['DELETE'])
def delete_prediction(id):
    if 'user' not in session:
        return jsonify({'status': 'error'}), 403

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('DELETE FROM predictions WHERE id=? AND username=?',
                   (id, session['user']))

    conn.commit()
    conn.close()

    return jsonify({'status': 'success'})
@app.route('/edit_prediction/<int:id>', methods=['PUT'])
def edit_prediction(id):
    if 'user' not in session:
        return jsonify({'status': 'error'}), 403

    data = request.get_json()

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('''
    UPDATE predictions
    SET age=?, stage=?, histology=?
    WHERE id=? AND username=?
    ''', (
        data['age'],
        data['stage'],
        data['histology'],
        id,
        session['user']
    ))

    conn.commit()
    conn.close()

    return jsonify({'status': 'success'})
# -------- LOGOUT --------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get('message', '').lower()

    # simple AI logic
    if "hello" in msg or "hi" in msg:
        reply = "Hello! I am your AI assistant. How can I help you?"

    elif "risk" in msg:
        reply = "Risk depends on survival probability: >70% Low, 40–70% Medium, <40% High."

    elif "stage" in msg:
        reply = "Cancer stages indicate severity. Stage I is early, Stage III is advanced."

    elif "survival" in msg:
        reply = "Survival probability shows chances of recovery. Higher is better."

    elif "treatment" in msg:
        reply = "Treatment depends on stage and patient condition. Consult a doctor for exact advice."

    elif "predict" in msg:
        reply = "Use the prediction form above to analyze patient data."

    else:
        reply = "I can help with risk, survival, cancer stages, and predictions."

    return jsonify({"reply": reply})
# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True)    