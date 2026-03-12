import streamlit as st
import pickle
import numpy as np

st.title("Credit Card Default Prediction")

st.write("Enter customer details")

LIMIT_BAL = st.number_input("Credit Limit")
AGE = st.number_input("Age")
PAY_0 = st.number_input("Repayment Status")
BILL_AMT1 = st.number_input("Bill Amount")
PAY_AMT1 = st.number_input("Payment Amount")

if st.button("Predict"):
    
    model = pickle.load(open("model.pkl","rb"))
    
    input_data = np.array([[LIMIT_BAL, AGE, PAY_0, BILL_AMT1, PAY_AMT1]])
    
    prediction = model.predict(input_data)

    if prediction == 1:
        st.error("Customer likely to DEFAULT")
    else:
        st.success("Customer NOT likely to default")
