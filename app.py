import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('spam_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Create a Streamlit application title
st.title("Spam Email Detector")

# Create a text area for user input
message = st.text_area("Enter the email message here:")

# Create a button to trigger prediction
if st.button("Predict"):
    # Vectorize the user's input message
    message_vec = vectorizer.transform([message])

    # Make a prediction
    prediction = model.predict(message_vec)

    # Display the prediction
    if prediction[0] == 1:
        st.write("Prediction: Spam")
    else:
        st.write("Prediction: Ham")
