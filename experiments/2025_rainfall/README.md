## KAGGLE TABULAR PLAYGROUND SERIES - RAINFALL PREDICTION
Walter Reade and Elizabeth Park. Binary Prediction with a Rainfall Dataset. https://kaggle.com/competitions/playground-series-s5e3, 2025. Kaggle.

## SPECIFICATIONS
- **Type:** Binary Classification
- **Target Variable:** `rainfall` (binary)
- **Evaluation Metric:** AUC-ROC
- **Data Source:** Synthetic data generated from model trained on "Rainfall Prediction using Machine Learning" dataset

## DATASET DETAILS
### train.csv
- **Features**:
  - `id`: Unique identifier
  - `day`: Day information
  - `pressure`: Atmospheric pressure
  - `maxtemp`: Maximum temperature
  - `temperature`: Current temperature
  - `mintemp`: Minimum temperature
  - `dewpoint`: Dew point temperature
  - `humidity`: Relative humidity percentage
  - `cloud`: Cloud coverage
  - `sunshine`: Sunshine duration/intensity
  - `winddirection`: Direction of wind
  - `windspeed`: Speed of wind
- **Target**: `rainfall` (binary outcome)

## TECHNICAL CONSIDERATIONS
- Feature distributions in synthetic data approximate but do not exactly match original dataset
- Dual-dataset approach: combining synthetic competition data with original dataset may improve performance (either as extra rows or extra columns)
- Competition designed for rapid experimentation with models and feature engineering techniques
