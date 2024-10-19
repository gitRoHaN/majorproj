import joblib
import pandas as pd
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
def cleaning(text):
    clean_text = text.lower()
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))

    return ' '.join(sentence)

def predict_mpg(text):
    pkl_filename = "ml_model1.pkl"
    with open(pkl_filename, 'rb') as f_in:
        ml = joblib.load(f_in)
    pkl_tfidf = "tfidf1.pkl"
    with open(pkl_tfidf, 'rb') as tf_in:
        tf = joblib.load(tf_in)
    clean_text = cleaning(text)
    tfid_matrix = tf.transform([clean_text])
    pred_proba = ml.predict_proba(tfid_matrix)
    idx = np.argmax(pred_proba)
    pred = ml.classes_[idx]
    return pred