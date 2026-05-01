from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = 'your_secret_key' # Required for sessions

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

@app.route('/predict', methods=['POST'])
def predict():
    # Placeholder for your Scikit-learn model prediction logic
    # You can integrate the clinical feature set models you've worked on here
    return jsonify({"prediction": "High Survival Probability"})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)