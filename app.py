from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and preprocessor
model = joblib.load("smart_canteen_model.pkl")
preprocessor = joblib.load("smart_canteen_preprocessor.pkl")

# Load the Smart Canteen dataset
dataset = pd.read_excel("Smart_Canteen_Final_Dataset.xlsx")


# -------------------------------------------------
# GRAPH DATA
# -------------------------------------------------

# Average demand by food item
demand_by_item = (
    dataset.groupby("Item")["Qty_Sold"]
    .mean()
    .round(0)
)

# Average demand by time slot
demand_by_time = (
    dataset.groupby("Time_Slot")["Qty_Sold"]
    .mean()
    .round(0)
)

# Average waste by food item
waste_by_item = (
    dataset.groupby("Item")["Qty_Left"]
    .mean()
    .round(0)
)


@app.route("/")
def home():

    return render_template(
        "index.html",

        # Graph 1
        item_labels=demand_by_item.index.tolist(),
        item_demand=demand_by_item.values.tolist(),

        # Graph 2
        time_labels=demand_by_time.index.tolist(),
        time_demand=demand_by_time.values.tolist(),

        # Graph 3
        waste_labels=waste_by_item.index.tolist(),
        item_waste=waste_by_item.values.tolist(),

        # No prediction shown initially
        prediction=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    # Get data from website form
    item = request.form["Item"]
    time_slot = request.form["Time_Slot"]
    day = request.form["Day"]
    weather = request.form["Weather"]
    week_type = request.form["Week_Type"]
    college_event = request.form["College_Event"]
    holiday = request.form["Holiday"]
    price = float(request.form["Price"])
    month = int(request.form["Month"])
    day_of_week_num = int(request.form["Day_of_Week_Num"])

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Item": item,
        "Time_Slot": time_slot,
        "Day": day,
        "Weather": weather,
        "Week_Type": week_type,
        "College_Event": college_event,
        "Holiday": holiday,
        "Price": price,
        "Month": month,
        "Day_of_Week_Num": day_of_week_num
    }])

    # Apply preprocessing
    input_processed = preprocessor.transform(input_data)

    # Predict food demand
    predicted_demand = model.predict(input_processed)[0]
    predicted_demand = round(predicted_demand)

    # Add 5% safety margin
    recommended_preparation = round(predicted_demand * 1.05)

    # Calculate estimated leftover
    estimated_leftover = (
        recommended_preparation - predicted_demand
    )

    # Calculate estimated waste percentage
    estimated_waste = round(
        (estimated_leftover / recommended_preparation) * 100,
        2
    )

    return render_template(
        "index.html",

        prediction=predicted_demand,
        preparation=recommended_preparation,
        leftover=estimated_leftover,
        waste=estimated_waste,

        # Graph 1
        item_labels=demand_by_item.index.tolist(),
        item_demand=demand_by_item.values.tolist(),

        # Graph 2
        time_labels=demand_by_time.index.tolist(),
        time_demand=demand_by_time.values.tolist(),

        # Graph 3
        waste_labels=waste_by_item.index.tolist(),
        item_waste=waste_by_item.values.tolist()
    )


if __name__ == "__main__":
    app.run(debug=True)