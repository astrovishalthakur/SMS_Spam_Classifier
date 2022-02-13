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
        
        
        
footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}

#love {
color: red;
}
</style>
<div class="footer">
<p>Developed with <span id="love">❤</span> by <a style='display: block; text-align: center;' href="https://github.com/astrovishalthakur" target="_blank">Vishal Thakur</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
