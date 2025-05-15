import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = 'kerala_datasets.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# List of columns to predict (excluding 'FLOODS' and 'YEAR')
columns_to_predict = [col for col in df.columns if col not in ['FLOODS', 'YEAR']]

# Ask the user for the number of future steps
future_steps = int(input("Enter the number of future steps to predict: "))

# Prepare to store predictions
predictions = {}

# Predict future values for each column
for column in columns_to_predict:
    try:
        # Ensure the column is numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # Fill missing values with forward-fill method
        df[column].fillna(method='ffill', inplace=True)

        # Fit ARIMA model
        model = ARIMA(df[column], order=(5, 1, 0))  # Example ARIMA order (5,1,0); can be tuned
        model_fit = model.fit()

        # Forecast the next values
        next_values = model_fit.forecast(steps=future_steps).tolist()
        predictions[column] = next_values[-1]
    except Exception as e:
        predictions[column] = f"Error: {e}"

# Display the predictions
print(f"\nPredicted next {future_steps} values for each column:")
for column, values in predictions.items():
    print(f"{column}: {values}")