import os
os.system('pip install scikit-learn numpy pandas streamlit')

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to perform linear regression and calculate error
def perform_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return model, mse

def main():
    st.title("Linear Regression App")

    # File uploader for input CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with 'x' and 'y' columns:", type="csv")

    if uploaded_file is not None:
        try:
            # Load the uploaded CSV file
            data = pd.read_csv(uploaded_file)

            # Check if required columns are present
            if 'x' not in data.columns or 'y' not in data.columns:
                st.error("The uploaded file must contain 'x' and 'y' columns.")
                return

            # Extract x and y values
            X = data['x'].values.reshape(-1, 1)
            y = data['y'].values

            # Perform linear regression
            model, mse = perform_linear_regression(X, y)

            # Display results
            st.success("Linear regression performed successfully!")
            st.write(f"Model Coefficient (Slope): {model.coef_[0]:.4f}")
            st.write(f"Model Intercept: {model.intercept_:.4f}")
            st.write(f"Mean Squared Error: {mse:.4f}")

            # Display predictions
            predictions = model.predict(X)
            results_df = pd.DataFrame({"x": X.flatten(), "y": y, "Predicted y": predictions})
            st.write("Predictions:")
            st.dataframe(results_df)

            # Plot results
            st.line_chart(pd.DataFrame({"Actual y": y, "Predicted y": predictions}, index=X.flatten()))
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()
