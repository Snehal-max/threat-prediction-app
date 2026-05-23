import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model

# Load model
model = load_model("model.h5")

# Sensor columns
sensors = [f"sensor_{i}" for i in range(25)]

# Title
st.title("Threat Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Function to prepare sequences
def prepare_data(df):

    sequences = []

    for seq_id, group in df.groupby("sequence_id"):

        group = group.sort_values("timestep")

        sequences.append(
            group[sensors].values
        )

    return np.array(sequences)

if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")
    st.dataframe(data.head())

    # Prepare data correctly
    X = prepare_data(data)

    # Predict
    predictions = model.predict(X)

    # Final class
    predicted_class = np.argmax(predictions, axis=1)

    # Sequence IDs
    sequence_ids = data.groupby("sequence_id").first().index.tolist()

    # Result dataframe
    result = pd.DataFrame({
        "sequence_id": sequence_ids,
        "Prediction": predicted_class
    })

    # Show results
    st.write("Prediction Results")
    st.dataframe(result)    