import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model = load_model('flower_classifier_model.keras')
class_names = ['Tulip', 'Orchid', 'Sunflower', 'Lotus', 'Lily']
st.title('Flower Classifier')
st.write('Upload a flower image to get the classification.')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    
    st.write(f"Prediction: {predicted_class}")