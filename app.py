from flask import Flask, render_template, url_for, request, redirect, jsonify
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import io
import base64


app = Flask (__name__)

path_base = os.path.join('src', 'model.pkl')

@app.route ('/')
@app.route ('/inicio') # Decorador

def inicio (): # función o vista
    return render_template ('index.html')

@app.route('/data')
def data ():
    df = pd.read_csv('data/wines_dataset.csv', sep = "|")
    sample_data = df[['class', 'volatile acidity', 'citric acid', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']].head(10) 
    sample_data_dict = sample_data.to_dict(orient='records')   
    return render_template('data.html', sample_data=sample_data_dict)

@app.route('/modelo')
def modelo():
    return render_template('modelo.html')

@app.route('/predicciones', methods=['GET', 'POST'])
def predicciones():
    if request.method == 'GET':
        # Renderiza la plantilla si la solicitud es GET
        return render_template('predicciones.html')

    if request.method == 'POST':
        try:
            # Maneja la solicitud POST para predicción
            #model = pickle.load(open(path_base + 'ad_model.pkl', 'rb'))
            model = pickle.load(open('ad_model.pkl', 'rb'))

            # Obtiene los valores del formulario
            wine_class = request.form.get('class', None)
            volatile_acidity = request.form.get('volatile_acidity', None)
            citric_acid = request.form.get('citric_acid', None)
            chlorides = request.form.get('chlorides', None)
            free_sulfur_dioxide = request.form.get('free_sulfur_dioxide', None)
            total_sulfur_dioxide = request.form.get('total_sulfur_dioxide', None)
            density = request.form.get('density', None)
            ph = request.form.get('ph', None)
            sulphates = request.form.get('sulphates', None)
            alcohol = request.form.get('alcohol', None)

            # Verifica si alguno de los campos está vacío
            if any(value is None for value in [wine_class, volatile_acidity, citric_acid, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]):
                return jsonify({"error": "Args empty, the data are not enough to predict"}), 400

            # Convierte los valores a float y usa los nombres de características correctos
            features = {
                'class': wine_class,
                'volatile acidity': float(volatile_acidity),
                'citric acid': float(citric_acid),
                'chlorides': float(chlorides),
                'free sulfur dioxide': float(free_sulfur_dioxide),
                'total sulfur dioxide': float(total_sulfur_dioxide),
                'density': float(density),
                'pH': float(ph),
                'sulphates': float(sulphates),
                'alcohol': float(alcohol)
            }
            
            # Convierte 'white' y 'red' a valores numéricos y añade al DataFrame
            if wine_class.lower() == 'white':
                features['class'] = 1
            elif wine_class.lower() == 'red':
                features['class'] = 0
            else:
                return jsonify({"error": "Invalid value for class"}), 400
            
            # Crea un DataFrame con los nombres de las características
            input_data = pd.DataFrame([features])

            # Realiza la predicción
            prediction = model.predict(input_data)
            prediction = int(prediction[0])
            return jsonify({'predictions': prediction})

        except Exception as e:
            return jsonify({"error": str(e)}), 500
                        
@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'GET':
        return render_template('retrain.html')

    if request.method == 'POST':
        if os.path.exists("./data/vinos_new.csv"):
            # Cargar y preparar el nuevo dataset
            data = pd.read_csv("./data/vinos_new.csv")
            X_train, X_test, y_train, y_test = train_test_split(
                data.drop(columns=['quality']),
                data['quality'],
                test_size=0.20,
                random_state=52
            )

            # Aplicar las mismas transformaciones
            X_train["class"] = (X_train["class"] == "white").astype(int)
            X_test["class"] = (X_test["class"] == "white").astype(int)

            features_to_transform = ["chlorides", "free sulfur dioxide", "total sulfur dioxide"]
            for col in features_to_transform:
                desplaza = 0
                if X_train[col].min() <= 0:
                    desplaza = int(abs(X_train[col].min())) + 1
                X_train[col] = np.log(X_train[col] + desplaza)
                X_test[col] = np.log(X_test[col] + desplaza)

            features_clf = ['class', 'volatile acidity', 'citric acid', 'chlorides',
                            'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol']

            # Retrain del modelo
            lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            param_grid = {
                'max_iter': [1000, 10000],
                'class_weight': ['balanced', False],
            }
            lr_grid = GridSearchCV(lr_clf, param_grid=param_grid, cv=5, scoring="balanced_accuracy")
            lr_grid.fit(X_train[features_clf], y_train)

            # Evaluar el nuevo modelo
            y_pred = lr_grid.best_estimator_.predict(X_test[features_clf])
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Generar matriz de confusión
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true", ax=ax)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)

            # Guardar el nuevo modelo entrenado
            with open('ad_model.pkl', 'wb') as file:
                pickle.dump(lr_grid.best_estimator_, file)

            return render_template('retrain_result.html', class_report=class_report, img_data=img_base64)

        else:
            return render_template('retrain_result.html', error="New data for retrain NOT FOUND. Nothing done!")

    

if __name__ == "__main__":
    os.environ ['FLASK_ENV'] = 'development'
    app.run (debug=True, host='0.0.0.0', port=os.getenv ('PORT', default=5000))
    