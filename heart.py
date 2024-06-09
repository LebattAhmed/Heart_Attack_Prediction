from pyexpat import features
from flask import Flask, render_template, request
import pickle
import pandas as pd


# The line `app = Flask(__name__)` in the provided Python code snippet is creating an instance of the
# Flask class, which represents the Flask application. The `__name__` argument is a special Python
# variable that represents the name of the current module. When Flask is initialized with `__name__`,
# it knows where to look for templates, static files, and other resources related to the application.
# This line essentially sets up the Flask application with the name of the current module.
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

# Charger le modèle SVM
# The code snippet `with open('tree_model.txt', 'rb') as model_file: model = pickle.load(model_file)`
# is loading a machine learning model from a file named 'tree_model.txt' using Python's `pickle`
# module.
with open('tree_model.txt', 'rb') as model_file:
    model = pickle.load(model_file)


# Créer un DataFrame avec une seule ligne pour obtenir les noms des colonnes
example_data = pd.DataFrame([[0]*13], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

# Définir la route principale
@app.route('/')
def index():
    return render_template('index.html')

# Définir la route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:

         # Convertir les valeurs du formulaire en types appropriés
        # This line of code is creating a dictionary called `features` by iterating over the columns
        # of the `example_data` DataFrame. For each column, it is checking if the column name is
        # 'oldpeak'. If it is 'oldpeak', it converts the corresponding form value to a float using
        # `float(request.form[col])`, otherwise, it converts it to an integer using
        # `int(request.form[col])`. The resulting key-value pairs in the `features` dictionary will
        # have column names as keys and the converted form values as values.
        features = {col: float(request.form[col]) if col == 'oldpeak' else int(request.form[col]) for col in example_data.columns}

        # Créer un DataFrame avec la seule ligne contenant les valeurs du formulaire
        input_data = pd.DataFrame([features])

        # Votre logique de prédiction ici
        prediction = model.predict(input_data)  # Assurez-vous de définir 'features' correctement

        # Reste du code pour la redirection ou le rendu de la page de résultat
        return render_template('result.html', prediction=prediction)

    except ValueError as ve:
          return render_template('error.html', error_message=f"ValueError: {ve}")


# Votre route pour la page d'erreur
@app.route('/error')
def error():
    return render_template('error.html', error_message="An error occurred.")

# Exécuter l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
