import streamlit as st
from PIL import Image
import pickle
import tensorflow
from keras.preprocessing import image
from keras.models import load_model
import numpy as np


model = load_model('model.h5')

def prediction(path, model, class_names):
    img = Image.open(path).convert('RGB') 
    img = img.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, axis = 0)
    img = img/255

    y_prob = model.predict(img)[0][0]
    print(f'Probability: {y_prob}')
    class_idx = (y_prob > 0.5).astype('int')
    class_name = class_names[class_idx]

    return class_name

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", 'png'])
classes = ['with mask', 'without mask']

# class_output = prediction(uploaded_file, model, classes)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    class_output = prediction(uploaded_file, model, classes)
    # st.header("Person is " + class_output)
    st.success(f"Person is {class_output}")
    
