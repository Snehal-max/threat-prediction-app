# 🌊 Leviathan Threat Classifier — Streamlit App

## File Structure
```
leviathan_app/
├── app.py            ← Streamlit frontend
├── model.py          ← Model architecture + prediction logic
├── requirements.txt  ← Dependencies
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Save your trained model weights from Kaggle
Add this at the end of your Kaggle notebook after training:
```python
model.save_weights('v5_weights.h5')
```
Then download `v5_weights.h5` from Kaggle outputs.

## App Features

### Single Prediction tab
- Enter sensor values manually for all 10 timesteps
- Or generate random demo data
- See threat level with color-coded result card
- See probability bars for all 4 classes
- See sensor activity heatmap

### Batch Prediction tab  
- Upload test.csv (same format as Kaggle)
- Classifies all sequences at once
- Shows prediction distribution chart
- Download submission.csv directly

### Sidebar
- Upload model weights (.h5)
- Upload train.csv to fit the scaler properly
- Shows model info

## Threat Levels
| Level | Name | Description |
|-------|------|-------------|
| 0 | DORMANT | Inactive, no threat |
| 1 | REACTIVE | Minor instability |
| 2 | PREDATORY | Aggressive behavior |
| 3 | EXTINCTION-LEVEL | Catastrophic event |
