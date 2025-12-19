Diabetes-Prediction
 Diabetes Prediction System

This project is a machine learning–based Diabetes Prediction System developed using Python and scikit-learn.  
It predicts whether a person is likely to have diabetes based on medical attributes provided by the user.

 Project Overview

The system uses a Random Forest Classifier to analyze health-related features such as glucose level, BMI, age, etc., and predicts the presence or absence of diabetes.  
The model is trained on a diabetes dataset and allows real-time prediction through user input.

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning (Random Forest)

 Dataset Features

The dataset includes the following attributes:

- Pregnancies  
- Glucose  
- Blood Pressure  
- Insulin  
- Body Mass Index (BMI)  
- Age  
- Outcome (0 = No Diabetes, 1 = Diabetes)

 Methodology

1. Load and preprocess the dataset  
2. Split data into training and testing sets  
3. Apply feature scaling using StandardScaler  
4. Train the model using Random Forest Classifier  
5. Evaluate the model using accuracy and classification report  
6. Predict diabetes based on user-provided input  

How to Run the Project

 Step 1: Install required libraries
pip install pandas scikit-learn
