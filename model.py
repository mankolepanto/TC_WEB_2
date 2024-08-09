import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

import pickle
import os


# Obtén el directorio actual de trabajo
base_dir = os.getcwd()

# Construye la ruta relativa al archivo
file_path = os.path.join(base_dir, "data", "wines_dataset.csv")

# Lee el archivo CSV
data = pd.read_csv(file_path, sep="|")

#Particion en train-test:
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['quality']),
                                                    data['quality'],
                                                    test_size = 0.20,
                                                    random_state=42)

#Tratamiento features:
X_train["class"] = (X_train["class"] == "white").astype(int) # white -> clase 1, red -> clase 0
X_test["class"] = (X_test["class"] == "white").astype(int) # white -> clase 1, red -> clase 0

features_to_transform = ["chlorides","free sulfur dioxide", "total sulfur dioxide"]
for col in features_to_transform:
    desplaza = 0 
    if X_train[col].min() <= 0:
        desplaza = int(abs(X_train[col].min())) + 1
    X_train[col] = np.log(X_train[col] + desplaza)
    X_test[col] = np.log(X_test[col] + desplaza)

features_clf = ['class', 'volatile acidity', 'citric acid', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

#Modelo de Regresión Logistica:
lr_clf = LogisticRegression(max_iter = 1000, class_weight = "balanced")

param_grid = {
    'max_iter':[1000,10000],
    'class_weight':['balanced',False],
    
}

lr_grid = GridSearchCV(lr_clf,
                        param_grid= param_grid,
                        cv = 5,
                        scoring = "balanced_accuracy")

lr_grid.fit(X_train[features_clf], y_train)

y_pred = lr_grid.best_estimator_.predict(X_test[features_clf])
class_report = classification_report(y_test, y_pred)
conf_matrix = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize = "true")

#Guardamos el modelo:
with open('ad_model.pkl', 'wb') as file:
    pickle.dump(lr_grid.best_estimator_, file)
