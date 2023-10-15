import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def main():

    st.title("Crab and Lobster Image Classifier")
    st.write("This app classifies whether the image is a Crab or a Lobster")

    @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model('weights-improvement-06-0.97.hdf5')
        return model

    def import_and_predict(image_data, model):
        size = (128, 128)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        image = np.asarray(image)
        image = image / 255.0
        img_reshape = np.reshape(image, (1, 128, 128, 3))
        prediction = model.predict(img_reshape)
        return prediction

    def display_prediction(image, model, class_names):
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        probability = np.max(prediction)
        string = f"Prediction: {class_name} | Probability: {probability:.2f}"
        st.success(string)

    model = load_model()
    class_names = ["Crab", "Lobster"]

    file = st.file_uploader("Select an image of Crab or Lobster on your computer", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "Prediction: " + class_name
        st.success(string)
 
if __name__ == "__main__":
    main()
