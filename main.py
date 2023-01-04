import streamlit as st
import json 
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from Data_class import Extraction
import random
import pickle
st.set_page_config(
   page_title="Pharmacy virtual Assistant",
   page_icon="💉",
   layout="wide",
   initial_sidebar_state="expanded",
)
st.title("Virtual Pharmacy Assistant 🖥️")
with open("intents.json") as file:
    data = json.load(file)
image = Image.open('logo_200.png')

st.image(image)

Liste_quariers=['al-fida','sidi-moumen','ain-chock','mers-sultan','sidi-bernoussi','ain-sebaa','lissasfa','bourgogne','polo','sidi-maarouf','roches-noires','maarif','al-azhar-panorama','quartier-des-hopitaux','hay-mohammadi','annasi','sidi-othmane']

input_text = st.text_input('Talk with me 👋🏻!')
def chat():
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    inp = input_text
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                            truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    if tag in Liste_quariers:
        index=Liste_quariers.index(tag)
        objet=Extraction(Liste_quariers[index])
        data1=objet.donnees()
        n=len(data1)
        for e in range(n):
                d=' '+data1['pharmacie'][e]+" ,in "+data1['adress'][e]+" to contact: "+data1['telephone'][e]
                original_text = '<p style="font-family:cursive; color:#31333F; font-size: 15px;">{}</p>'.format(' 💬   '+d)
                st.markdown(original_text,unsafe_allow_html=True)
    else:
        for i in data['intents']:
            if i['tag'] == tag:
                    original_text = '<p style="font-family:cursive; color:#31333F;border-left: thick double #32a1ce; font-size: 15px;">{}</p>'.format(' 💬    '+np.random.choice(i['responses']))
                    st.write(original_text,unsafe_allow_html=True)

    

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))
chat()