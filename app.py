from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained pipeline model
model = joblib.load("flight_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        airline = request.form["airline"]
        source_city = request.form["source_city"]
        destination_city = request.form["destination_city"]
        departure_time = request.form["departure_time"]
        arrival_time = request.form["arrival_time"]

        stops = int(request.form["stops"])
        travel_class = int(request.form["class"])
        duration = float(request.form["duration"])
        days_left = int(request.form["days_left"])

        # Create dataframe for prediction
        input_data = pd.DataFrame({
            "airline":[airline],
            "source_city":[source_city],
            "destination_city":[destination_city],
            "departure_time":[departure_time],
            "arrival_time":[arrival_time],
            "stops":[stops],
            "class":[travel_class],
            "duration":[duration],
            "days_left":[days_left]
        })

        prediction = model.predict(input_data)

        output = round(prediction[0],2)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Flight Price: ₹ {output}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)