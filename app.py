import numpy as np
import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize


stwd = np.load('stop_words_array.npy')
with open('vectorizer.pkl', 'rb') as f:
    vec = pickle.load(f)

with open('nb_model2500.pkl', 'rb') as f:
    model = pickle.load(f)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

def clean_text(text):
    lemma = WordNetLemmatizer()
    token=word_tokenize(text.lower()) #case conversion + tokenization
  #non-alpha removal
    ftoken=[i for i in token if i.isalpha()]
  #stopwords removal
    stoken=[i for i in ftoken if i not in stwd]
  #lemma
    ltoken=[lemma.lemmatize(i) for i in stoken]
  #joining list of msgs
    x =  " ".join(ltoken)
    inpute = vec.transform([x]).toarray()
    return model.predict(inpute)
# print(clean_text('hii my name is shaukat.'))


# Streamlit App
st.title("Sentiment Analysis App 😊😞")
st.write("Enter a review to predict its sentiment:")

user_input = st.text_area("Your Review:", "")
if st.button("Analyze Sentiment"):
    if user_input:
        prediction = clean_text(user_input)
        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😞")
