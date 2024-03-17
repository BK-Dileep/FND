# app1.py

import streamlit as st
from model import predict_fake_news

def main():
    st.title("Fake News Detection App")
    st.write("Enter the text you want to analyze:")

    # Text input box for user input
    user_input = st.text_area("Input Text", "")

    # Button to trigger prediction
    if st.button("Predict"):
        if user_input.strip() != "":
            prediction = predict_fake_news(user_input)
            if prediction == 1:
                st.error("This news might be real.")
            else:
                st.error("This news might be fake.")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
