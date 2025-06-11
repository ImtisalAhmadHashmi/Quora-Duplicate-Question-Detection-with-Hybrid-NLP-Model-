import streamlit as st
from helper import duplicate_prediction


st.title("Duplicate Question Detector")

q1 = st.text_input("Enter first question:")
q2 = st.text_input("Enter second question:")

if st.button("Check Duplicate"):

    prediction = duplicate_prediction(q1, q2)

    st.write(f"Probability of being duplicate: {prediction[0][0]:.2f}")
    st.write(f"Verdict: {'DUPLICATE' if prediction > 0.5 else 'NOT DUPLICATE'}")