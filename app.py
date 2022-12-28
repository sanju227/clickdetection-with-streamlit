import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('clickbait_data.csv')
text = df['headline'].values
labels = df['clickbait'].values
text_train, text_test, y_train, y_test = train_test_split(text, labels, test_size=0.2)
vocab_size = 5000
maxlen = 500
embedding_size = 32
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(text)
model=tf.keras.models.load_model('model.h5')
st.write("""
         # Clickbait Detection
         """
         )
t = st.text_input("please write text")
test=[t]
def import_and_predict(test, model):
    token_text = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=maxlen)
    preds = [round(i[0]) for i in model.predict(token_text)]
    for (text, pred) in zip(test, preds):
        label = 'Clickbait' if pred == 1.0 else 'Not Clickbait'
        prediction=("{} - {}".format(text, label))
    return prediction
if test is None:
    st.text("Please upload the text")
else:
    predictions = import_and_predict(test, model) 
    st.success(predictions)
