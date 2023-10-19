import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def main():
    # Set up the Streamlit app

    st.title("Crab and Lobster Image Classifier with Rating")
    st.write("This app classifies whether the image is a Crab or a Lobster and allows you to rate the prediction")

    @st.cache(allow_output_mutation=True)


        rating = st.slider(f"Rate the prediction {index + 1}", 1, 5, 3, key=f"slider_{index}")
        st.write(f"You rated the prediction {index + 1}: {rating} stars")

    model = load_model()
    class_names = ["Crab", "Lobster"]

    file_list = st.file_uploader("Select multiple images of Crab or Lobster in your computer", accept_multiple_files=True)

    if not file_list:
        st.text("Please upload one or more image files")
    else:
        for i, file in enumerate(file_list):
            image = Image.open(file)
            display_prediction(image, model, class_names, i)

if __name__ == "__main__":
    main()
