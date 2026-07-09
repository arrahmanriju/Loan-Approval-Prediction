from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBClassifier

app = Flask(__name__)

model = XGBClassifier()
model.load_model("model.json")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ==========================
        # Numerical Features
        # ==========================
        age = int(request.form["age"])
        income = float(request.form["income"])
        experience = int(request.form["experience"])
        loan_amount = float(request.form["loan_amount"])
        interest_rate = float(request.form["interest_rate"])
        loan_percentage = float(request.form["loan_percentage"])
        credit_history = int(request.form["credit_history"])
        credit_score = int(request.form["credit_score"])

        # ==========================
        # Label Encoding
        # ==========================
        gender = 1 if request.form["gender"] == "male" else 0
        previous_loan = 1 if request.form["previous_loan"] == "Yes" else 0

        # ==========================
        # Initialize all one-hot columns with 0
        # ==========================
        data = {
            "Age": age,
            "Gender": gender,
            "Person Income": income,
            "Employee Experience": experience,
            "Loan Amount": loan_amount,
            "Loan interest Rate": interest_rate,
            "Loan percentage": loan_percentage,
            "Credit History": credit_history,
            "Credit Score": credit_score,
            "Previous Loan": previous_loan,

            "Education_Bachelor": 0,
            "Education_Doctorate": 0,
            "Education_High School": 0,
            "Education_Master": 0,

            "Home Onwership_OTHER": 0,
            "Home Onwership_OWN": 0,
            "Home Onwership_RENT": 0,

            "Loan Intent_EDUCATION": 0,
            "Loan Intent_HOMEIMPROVEMENT": 0,
            "Loan Intent_MEDICAL": 0,
            "Loan Intent_PERSONAL": 0,
            "Loan Intent_VENTURE": 0
        }

        # ==========================
        # Education
        # Associate is dropped
        # ==========================
        education = request.form["education"]

        if education == "Bachelor":
            data["Education_Bachelor"] = 1
        elif education == "Doctorate":
            data["Education_Doctorate"] = 1
        elif education == "High School":
            data["Education_High School"] = 1
        elif education == "Master":
            data["Education_Master"] = 1

        # ==========================
        # Home Ownership
        # MORTGAGE is dropped
        # ==========================
        home = request.form["home"]

        if home == "OTHER":
            data["Home Onwership_OTHER"] = 1
        elif home == "OWN":
            data["Home Onwership_OWN"] = 1
        elif home == "RENT":
            data["Home Onwership_RENT"] = 1

        # ==========================
        # Loan Intent
        # DEBTCONSOLIDATION is dropped
        # ==========================
        intent = request.form["intent"]

        if intent == "EDUCATION":
            data["Loan Intent_EDUCATION"] = 1
        elif intent == "HOMEIMPROVEMENT":
            data["Loan Intent_HOMEIMPROVEMENT"] = 1
        elif intent == "MEDICAL":
            data["Loan Intent_MEDICAL"] = 1
        elif intent == "PERSONAL":
            data["Loan Intent_PERSONAL"] = 1
        elif intent == "VENTURE":
            data["Loan Intent_VENTURE"] = 1

        # ==========================
        # Create DataFrame
        # ==========================
        feature_order = [
            'Age',
            'Gender',
            'Person Income',
            'Employee Experience',
            'Loan Amount',
            'Loan interest Rate',
            'Loan percentage',
            'Credit History',
            'Credit Score',
            'Previous Loan',
            'Education_Bachelor',
            'Education_Doctorate',
            'Education_High School',
            'Education_Master',
            'Home Onwership_OTHER',
            'Home Onwership_OWN',
            'Home Onwership_RENT',
            'Loan Intent_EDUCATION',
            'Loan Intent_HOMEIMPROVEMENT',
            'Loan Intent_MEDICAL',
            'Loan Intent_PERSONAL',
            'Loan Intent_VENTURE'
        ]

        input_df = pd.DataFrame([data])
        input_df = input_df[feature_order]

        # ==========================
        # Prediction
        # ==========================
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            result = "Loan Approved"
        else:
            result = "Loan Rejected"

        # Optional probability
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = round(model.predict_proba(input_df)[0].max() * 100, 2)

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)