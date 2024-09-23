from flask import Flask, request, jsonify, render_template
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import pandas as pd

app = Flask(__name__)

# Load model
model = None
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# PostgreSQL connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'postgres'),
            database=os.getenv('DB_NAME', 'postgres'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres')
        )
        print("Database connection successful.")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = {
        'Sex_Code': [int(data['Sex_Code'])],
        'Pclass': [int(data['Pclass'])],
        'Embarked_Code': [int(data['Embarked_Code'])],
        'Title_Code': [int(data['Title_Code'])],
        'FamilySize': [int(data['FamilySize'])],
        'AgeBin_Code': [int(data['AgeBin_Code'])],
        'FareBin_Code': [int(data['FareBin_Code'])]
    }
    df_features = pd.DataFrame(features)
    print(f"Received features: {df_features}")
    
    if model is None:
        return jsonify({'error': 'Model is not loaded'})

    try:
        prediction = model.predict(df_features)[0]
        print(f"Prediction: {prediction}")
        prediction = int(prediction)  # Convert to standard Python integer
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

    # Save result
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO predictions (Sex_Code, Pclass, Embarked_Code, Title_Code, FamilySize, AgeBin_Code, FareBin_Code, prediction) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)',
                (data['Sex_Code'], data['Pclass'], data['Embarked_Code'], data['Title_Code'], data['FamilySize'], data['AgeBin_Code'], data['FareBin_Code'], prediction)
            )
            conn.commit()
            cursor.close()
            conn.close()
            print("Prediction saved to database.")
        except Exception as e:
            print(f"Error saving prediction to database: {e}")
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Database connection failed'})

    return render_template('index.html', prediction=prediction)


@app.route('/predictions', methods=['GET'])
def get_predictions():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('SELECT * FROM predictions')
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            return jsonify(results)
        except Exception as e:
            print(f"Error retrieving predictions from database: {e}")
            return jsonify({'error': str(e)})
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
