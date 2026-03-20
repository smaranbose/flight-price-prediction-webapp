import pandas as pd
import math
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# -----------------------------
# Data Loading
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(df):

    # Drop unnecessary columns
    df.drop(columns=['Unnamed: 0', 'flight'], errors='ignore', inplace=True)

    # Convert class to numeric
    df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)

    # Convert stops to numeric
    df['stops'] = pd.factorize(df['stops'])[0]

    return df


# -----------------------------
# Evaluate Model
# -----------------------------
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", math.sqrt(mean_squared_error(y_test, y_pred)))

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Flight Price")
    plt.ylabel("Predicted Flight Price")
    plt.title("Prediction vs Actual Price")
    plt.show()


# -----------------------------
# Main Pipeline
# -----------------------------
def main():

    path = "Clean_Dataset.csv"

    df = load_data(path)
    df = preprocess(df)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Categorical columns
    categorical_cols = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "arrival_time"
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    # Full ML pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_jobs=-1))
        ]
    )

    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate
    evaluate_model(pipeline, X_test, y_test)

    # Save pipeline
    joblib.dump(pipeline, "flight_price_model.pkl")

    print("Pipeline model saved as flight_price_model.pkl")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    main()

