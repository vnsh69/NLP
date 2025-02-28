import streamlit as st
import numpy as np 
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("next_word_predition.h5")
with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, input_text, tokenizer, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    if len(token_list)>=max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list)
    predicted_max_index = np.argmax(predicted,axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_max_index:
            return word
        
    return None 

st.title('next word prediction from sherlock homes book')
input_text = st.text_input('enter the sentence')
if st.button('predict'):
    max_len = model.input_shape[1]+1
    next_word = predict_next_word(model, input_text, tokenizer, max_len)
    st.write(f"next word is {next_word}")