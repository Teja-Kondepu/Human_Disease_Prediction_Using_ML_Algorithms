from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Set a secret key for the session

# List of Symptoms
all_symptoms = ['itching','skin_rash','back_pain','shivering','constipation','joint_pain','abdominal_pain','diarrhoea',
                'mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','swelling_of_stomach','vomiting',
                'swelled_lymph_nodes','blurred_and_distorted_vision','throat_irritation','continuous_sneezing',
                'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
                'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus',
                'neck_pain','dizziness','cramps','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
                'enlarged_thyroid','brittle_nails','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
                'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
                'spinning_movements','loss_of_balance','unsteadiness','patches_in_throat','weakness_of_one_body_side',
                'loss_of_smell','bladder_discomfort','continuous_feel_of_urine','passage_of_gases',
                'internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','red_spots_over_body',
                'belly_pain','abnormal_menstruation','watering_from_eyes','high_fever','headache','yellowish_skin',
                'lack_of_concentration','visual_disturbances','anxiety','cold_hands_and_feets','coma','stomach_bleeding',
                'pain_behind_the_eyes','dark_urine','dehydration','indigestion','history_of_alcohol_consumption',
                'fluid_overload','blood_in_sputum','weight_loss','painful_walking','pus_filled_pimples','blackheads',
                'scurring','skin_peeling','loss_of_appetite','silver_like_dusting','small_dents_in_nails','red_sore_around_nose',
                'irregular_sugar_level','yellow_crust_ooze']

# List of Diseases
diseases = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
            'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
            'Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox',
            'Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E',
            'Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
            'Heart attack','Varicose veins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
            'Arthritis','(vertigo) Paroymsal Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
            'Impetigo']

# Load datasets
df = pd.read_csv("datasets/Training.csv")
df.replace({'prognosis': {disease: disease for disease in diseases}}, inplace=True)

# Filter valid symptoms and sort them
valid_symptoms = sorted([symptom for symptom in all_symptoms if symptom in df.columns])

X = df[valid_symptoms]
y = df["prognosis"].astype(str)  # Ensure all values in y are strings

# Ensure y is categorical with discrete values
le = LabelEncoder()
y = le.fit_transform(y)

tr = pd.read_csv("datasets/Testing.csv")
tr.replace({'prognosis': {disease: disease for disease in diseases}}, inplace=True)

X_test = tr[valid_symptoms]
y_test = le.transform(tr["prognosis"].astype(str))

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/patient_info', methods=['GET', 'POST'])
def patient_info():
    if request.method == 'POST':
        # Save patient info in session or pass it to the next page
        session['patient_info'] = {
            'name': request.form['name'],
            'gender': request.form['gender'],
            'age': request.form['age'],
            'village': request.form['village']
        }
        return redirect(url_for('index'))
    return render_template('patient_info.html')

@app.route('/predictor')
def index():
    return render_template('index.html', symptoms=valid_symptoms, predictions={}, selected_symptoms=["", "", "", "", ""])

@app.route('/predict', methods=['POST'])
def predict():
    if 'reset' in request.form:
        return redirect(url_for('index'))
    elif 'exit' in request.form:
        return 'Exiting the system...'

    model_name = request.form.get('model', '').replace(" ", "_")
    if not model_name:
        return redirect(url_for('index'))

    symptoms_selected = [
        request.form.get('symptom1'),
        request.form.get('symptom2'),
        request.form.get('symptom3'),
        request.form.get('symptom4'),
        request.form.get('symptom5')
    ]

    inputtest = [0] * len(X.columns)
    for symptom in symptoms_selected:
        if symptom in X.columns:
            inputtest[X.columns.get_loc(symptom)] = 1

    if model_name == 'Decision_Tree':
        model = tree.DecisionTreeClassifier()
    elif model_name == 'Random_Forest':
        model = RandomForestClassifier()
    elif model_name == 'Naive_Bayes':
        model = GaussianNB()
    elif model_name == 'SVM':
        model = svm.SVC()

    model = model.fit(X, y)
    prediction = model.predict([inputtest])[0]
    disease = le.inverse_transform([prediction])[0]

    # Calculating accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy of {model_name.replace('_', ' ')}: {accuracy:.7f}")

    predictions = json.loads(request.form.get('predictions', '{}'))
    predictions[model_name] = disease

    return render_template('index.html', symptoms=valid_symptoms, predictions=predictions, selected_symptoms=symptoms_selected)

@app.route('/print_data')
def print_data():
    patient_info = session.get('patient_info', {})
    symptoms = request.args.get('symptoms', '').split(',')
    predictions = request.args.get('predictions', '{}')
    predictions = json.loads(predictions)
    return render_template('print_data.html', patient_info=patient_info, symptoms=symptoms, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)