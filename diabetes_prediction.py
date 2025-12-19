import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


file_path = "/content/diabetes.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'Outcome']

data = pd.read_csv(file_path, header=0, names=columns)


X = data.drop('Outcome', axis=1)
y = data['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


def predict_diabetes(input_data):

    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)

    if prediction == 1:
        return "The person is predicted to have diabetes."
    else:
        return "The person is predicted to not have diabetes."


print("\nEnter the following details to predict diabetes:")

try:
    pregnancies = int(input("Number of pregnancies: "))
    glucose = float(input("Glucose concentration: "))
    blood_pressure = float(input("Blood pressure (mm Hg): "))
    insulin = float(input("Insulin level (mu U/ml): "))
    bmi = float(input("Body Mass Index (BMI): "))
    age = int(input("Age (years): "))

    # Prepare input data for prediction
    user_input = [pregnancies, glucose, blood_pressure, insulin, bmi, age]

    # Predict the result based on user input
    prediction_result = predict_diabetes(user_input)
    print(f"\nPrediction: {prediction_result}")

except ValueError:
    print("Please enter valid numerical values for the inputs.")
