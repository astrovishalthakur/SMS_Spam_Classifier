import streamlit as st
import pickle
from textpreprocessor import transform_text

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

# 1. preprocess
transform_sms = transform_text(input_sms)

# 2. vectorized
vector = tfidf.transform([transform_sms])

# 3. predict
result = model.predict(vector)[0]

# 4. Display

if st.button("Classify"):

    if result == 1:
        st.header("Spam")

    elif result == 0:
        st.header("Not Spam")
    else:
        st.header("got error")
