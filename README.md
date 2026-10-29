# Used-Car Price Prediction (Flask + scikit-learn)

A web app that estimates the resale price of a used car from its make, model, year, fuel type and kilometers driven. A scikit-learn regression pipeline does the prediction, and a Flask backend serves it to a browser form that returns the estimate instantly.

## Screenshots

![App screenshot 1](https://github.com/pranaylachuluri/Car-prize-prediction/assets/103355762/a1d08d71-fd6d-4eb1-9070-de41d3bc2a46)
![App screenshot 2](https://github.com/pranaylachuluri/Car-prize-prediction/assets/103355762/3437cbf3-a1a7-45fd-b26d-eea87057725f)
![App screenshot 3](https://github.com/pranaylachuluri/Car-prize-prediction/assets/103355762/632bbbed-d9e8-4efc-ace1-6fac43f043e2)

## How it works

1. **Clean the data** (`Predictor.ipynb`): 892 raw Quikr car listings cleaned into `Cleaned Car.csv` (fixing prices, years, kilometers and names, removing outliers), leaving 815 cars.
2. **Build the pipeline** (`train.py`): one-hot encode the categorical columns (name, company, fuel type) with a `ColumnTransformer`, then fit `LinearRegression`, all in one scikit-learn `Pipeline`. Car names the model has never seen are handled instead of crashing.
3. **Evaluate**: R² of 0.64 ± 0.10 with 5-fold cross-validation (0.57 on a fixed 20% hold-out split, mean absolute error about ₹148k). The final model is refit on all the data.
4. **Serve** (`app.py`): Flask loads the trained pipeline from `LinearRegressionModel.pkl`, and retrains it automatically from the CSV if the file is missing or was saved by an incompatible scikit-learn version. The form in `templates/index.html` fills the model list from the chosen company and posts to `/predict`, which validates the input before predicting.

## Tech stack

Python · Pandas · scikit-learn · Flask · HTML / CSS / JavaScript

## Run it

```bash
pip install -r requirements.txt
python train.py   # optional: retrain and print the evaluation metrics
python app.py
```

Then open http://127.0.0.1:10000 in your browser.

## Deploy

The repo is ready to host for free on either platform:

- **Render:** New → Blueprint → pick this repo. `render.yaml` sets up the build and runs the app with gunicorn.
- **Hugging Face Spaces:** create a Space with the Docker SDK and push this repo to it. The `Dockerfile` serves the app on port 7860.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Flask app and `/predict` endpoint |
| `Predictor.ipynb` | Data exploration and cleaning |
| `train.py` | Reproducible training and evaluation |
| `Cleaned Car.csv` | Cleaned dataset |
| `LinearRegressionModel.pkl` | Trained pipeline |
| `templates/`, `static/` | Web form and styling |
| `render.yaml`, `Dockerfile` | Hosting configuration |
