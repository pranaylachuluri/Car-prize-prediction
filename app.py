import os
import pickle

import pandas as pd
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

from train import DATA_PATH, FEATURES, MODEL_PATH, train

app = Flask(__name__)
cors = CORS(app)
car = pd.read_csv(DATA_PATH, index_col=0)


def load_model():
    # Retrain when the saved model is missing or was pickled by an incompatible scikit-learn version
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                saved = pickle.load(f)
            saved.predict(car[FEATURES].head(1))
            return saved
        except Exception as e:
            print(f'Could not load {MODEL_PATH} ({e}), retraining')
    return train(verbose=False)


model = load_model()


@app.route('/', methods=['GET'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    fuel_type = request.form.get('fuel_type')

    if not company or company == 'Select Company' or not car_model:
        return 'Please select a company and a model.', 400
    try:
        year = int(request.form.get('year'))
        driven = int(float(request.form.get('kilo_driven')))
    except (TypeError, ValueError):
        return 'Please enter the kilometres driven as a number.', 400
    if driven < 0:
        return 'Kilometres driven cannot be negative.', 400

    features = pd.DataFrame([[car_model, company, year, driven, fuel_type]], columns=FEATURES)
    prediction = max(float(model.predict(features)[0]), 0.0)
    return f'{prediction:,.0f}'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
