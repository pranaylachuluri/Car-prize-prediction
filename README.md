# Used-Car Price Prediction (Flask + scikit-learn)

A web app that estimates the resale price of a used car from its make, model, year, fuel type and kilometers driven. A scikit-learn regression pipeline does the prediction, and a Flask backend serves it to a browser form that returns the estimate instantly.

## Screenshots

![App screenshot 1](https://github.com/pranaylachuluri/Car-prize-prediction/assets/103355762/a1d08d71-fd6d-4eb1-9070-de41d3bc2a46)
![App screenshot 2](https://github.com/pranaylachuluri/Car-prize-prediction/assets/103355762/3437cbf3-a1a7-45fd-b26d-eea87057725f)
![App screenshot 3](https://github.com/pranaylachuluri/Car-prize-prediction/assets/103355762/632bbbed-d9e8-4efc-ace1-6fac43f043e2)

## How it works

1. **Clean the data** (`Predictor.ipynb`): 892 raw Quikr car listings cleaned into `Cleaned Car.csv` (fixing prices, years, kilometers and names, removing outliers).
2. **Build the pipeline**: one-hot encode the categorical columns (name, company, fuel type) with a `ColumnTransformer`, then fit `LinearRegression`, all in one scikit-learn `Pipeline`.
3. **Evaluate**: R² of about 0.73 on a held-out 20% test split.
4. **Serve** (`app.py`): the trained pipeline is pickled to `LinearRegressionModel.pkl` and loaded by Flask. The form in `templates/index.html` fills the model list from the chosen company and posts to `/predict`.

## Tech stack

Python · Pandas · scikit-learn · Flask · HTML / CSS / JavaScript

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:10000 in your browser.

> `requirements.txt` pins older versions (scikit-learn 0.22, Flask 1.1) to match the saved model. Use a Python 3.7 or 3.8 virtual environment if the newest Python can't install them.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Flask app and `/predict` endpoint |
| `Predictor.ipynb` | Data cleaning, training and evaluation |
| `Cleaned Car.csv` | Cleaned dataset |
| `LinearRegressionModel.pkl` | Trained pipeline |
| `templates/`, `static/` | Web form and styling |
