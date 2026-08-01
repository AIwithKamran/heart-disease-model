# Heart Disease Prediction App

A Streamlit web app that predicts heart disease risk from patient inputs using a pre-trained KNN model.

## Project Files

- `app.py` — Streamlit application UI and prediction logic
- `knn_heart_model.pkl` — trained KNN classifier
- `heart_scaler.pkl` — fitted feature scaler
- `heart_columns.pkl` — expected model input columns
- `requirements.txt` — Python dependencies

## Prerequisites

- Python 3.10+
- `pip`

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

From the repository root:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## How It Works

1. Enter patient details in the UI.
2. The app builds a feature row and aligns it to the model's expected columns.
3. Features are scaled with the saved scaler.
4. The KNN model predicts risk:
   - `1` → High risk
   - `0` → Low risk

## Note

This tool is for educational/demo use only and is not a medical diagnosis system.
