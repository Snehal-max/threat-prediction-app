# Threat Prediction App

A simple Streamlit web app for predicting threat levels using a Deep Learning model.

---

## Features

- Upload CSV sensor data
- Predict threat levels
- View predictions in browser
- Download-ready prediction output
- Simple and lightweight UI

---

## Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn

---

## Project Structure

```text
project/
│
├── app.py
├── model.h5
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/threat-prediction-app.git
```

Move into the project folder:

```bash
cd threat-prediction-app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

If Streamlit command does not work:

```bash
python -m streamlit run app.py
```

---

## Input Format

Upload a CSV file containing:

- `sequence_id`
- `timestep`
- `sensor_0` to `sensor_24`

Each sequence should contain 10 timesteps.

---

## Model

The app uses a TensorFlow/Keras Conv1D model trained on sequential sensor data for multi-class threat prediction.

---

## Example Workflow

1. Start the Streamlit app
2. Upload test CSV file
3. Model predicts threat levels
4. Results displayed instantly

---

## Author

Snehal Muduli