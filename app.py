import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Animal Species Recognition",
    page_icon="🐾",
    layout="centered"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_animal_model.keras"
CLASS_PATH = BASE_DIR / "class_names.json"

# Display friendly English names
ENGLISH_NAMES = {
    "cane": "Dog",
    "cavallo": "Horse",
    "elefante": "Elephant",
    "farfalla": "Butterfly",
    "gallina": "Chicken",
    "gatto": "Cat",
    "mucca": "Cow",
    "pecora": "Sheep",
    "ragno": "Spider",
    "scoiattolo": "Squirrel"
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

st.title("🐾 Animal Species Recognition")
st.write(
    "Upload an animal photograph and let the trained "
    "deep learning model predict its species."
)

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded animal image",
        use_container_width=True
    )

    if st.button("Recognize Animal", type="primary"):
        with st.spinner("Analyzing image..."):
            # Prepare image exactly as during training
            resized = image.resize((224, 224))
            image_array = np.asarray(
                resized, dtype=np.float32
            )

            image_array = (
                tf.keras.applications.mobilenet_v2.preprocess_input(
                    image_array
                )
            )

            image_array = np.expand_dims(image_array, axis=0)

            model = load_model()
            class_names = load_classes()

            probabilities = model.predict(
                image_array, verbose=0
            )[0]

            top_indices = np.argsort(probabilities)[::-1][:3]
            best_index = int(top_indices[0])
            confidence = float(probabilities[best_index])

        predicted_class = class_names[best_index]
        animal_name = ENGLISH_NAMES.get(
            predicted_class, predicted_class
        )

        st.subheader("Prediction")
        st.success(f"Animal detected: {animal_name}")
        st.metric("Model confidence", f"{confidence * 100:.2f}%")

        if confidence < 0.60:
            st.warning(
                "The model is uncertain about this image. "
                "Try a clearer photograph."
            )

        st.subheader("Top 3 predictions")

        for index in top_indices:
            name = ENGLISH_NAMES.get(
                class_names[index], class_names[index]
            )
            score = float(probabilities[index])

            st.write(f"**{name}** — {score * 100:.2f}%")
            st.progress(score)

st.divider()
st.caption(
    "Model: MobileNetV2 transfer learning | "
    "Trained on 10 animal categories"
)
st.caption(
    "The model predicts only among its 10 trained categories. "
    "A high confidence score does not guarantee a correct prediction."
)
