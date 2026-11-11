import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Step 1: Load the dataset
data = pd.read_csv("clean_fuel.csv")

# Step 2: Create separate encoders for each column
class_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()

# Step 3: Fit and transform each categorical column
data['VEHICLE CLASS'] = class_encoder.fit_transform(data['VEHICLE CLASS'])
data['TRANSMISSION'] = transmission_encoder.fit_transform(data['TRANSMISSION'])
data['FUEL'] = fuel_encoder.fit_transform(data['FUEL'])
print(data)

# Step 4: Define input features and target columns
features = ["VEHICLE CLASS", "ENGINE SIZE", "CYLINDERS", "TRANSMISSION", "FUEL", "EMISSIONS"]
target = ["FUEL CONSUMPTION", "HWY (L/100 km)", "COMB (L/100 km)"]

X = data[features]
y = data[target]

# Step 5: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   


# Step 6: Train the Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate model
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print("R-squared score:", score)

user_input = {
    "VEHICLE CLASS": input(f"Enter vehicle class  {list(class_encoder.classes_)}: "),
    "ENGINE SIZE": float(input("Enter engine size: ")),
    "CYLINDERS": int(input("Enter cylinders: ")),
    "TRANSMISSION": input(f"Enter transmission {list(transmission_encoder.classes_)}: "),
    "FUEL": input(f"Enter fuel {list(fuel_encoder.classes_)}: "),
    "EMISSIONS": int(input("Enter emissions: "))
}

try:
    user_input["VEHICLE CLASS"] = class_encoder.transform([user_input["VEHICLE CLASS"]])[0]
    user_input["TRANSMISSION"] = transmission_encoder.transform([user_input["TRANSMISSION"]])[0]
    user_input["FUEL"] = fuel_encoder.transform([user_input["FUEL"]])[0]
except ValueError as e:
    print("\n Error: One of the inputs is not recognized.")
    exit()

# Step 10: Convert user input into DataFrame
user_df = pd.DataFrame(user_input, index=[0])

# Step 11: Predict for user input
user_predictions = model.predict(user_df)

# Step 12: Display predictions

print("\nPredicted values:")
for i, target_col in enumerate(target):
    print(f"{target_col}: {user_predictions[0][i]:.2f}")
