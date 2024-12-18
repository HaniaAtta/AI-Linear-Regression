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

    # Input for x values
    x_values = st.text_area(
        "Enter x values (comma-separated):",
        placeholder="e.g., 1, 2, 3, 4, 5",
    )

    # Input for y values
    y_values = st.text_area(
        "Enter y values (comma-separated):",
        placeholder="e.g., 2, 4, 6, 8, 10",
    )

    if st.button("Perform Linear Regression"):
        try:
            # Parse x and y values
            X = np.array([float(x) for x in x_values.split(",")]).reshape(-1, 1)
            y = np.array([float(y) for y in y_values.split(",")])

            if len(X) != len(y):
                st.error("The number of x and y values must be equal.")
            else:
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
        except ValueError:
            st.error("Please ensure all inputs are numeric and properly formatted.")

if __name__ == "__main__":
    main()

