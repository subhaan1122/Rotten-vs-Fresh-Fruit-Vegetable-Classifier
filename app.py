import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Load trained model
model = load_model("fruit_veg_model.keras")  # local model path

# ✅ Define class labels
class_labels = ['Apple__Healthy', 'Apple__Rotten', 'Banana__Healthy', 'Banana__Rotten',
                'Bittergourd__Healthy', 'Bittergourd__Rotten', 'Capsicum__Healthy', 'Capsicum__Rotten',
                'Cucumber__Healthy', 'Cucumber__Rotten', 'Okra__Healthy', 'Okra__Rotten',
                'Orange__Healthy', 'Orange__Rotten', 'Potato__Healthy', 'Potato__Rotten',
                'Tomato__Healthy', 'Tomato__Rotten', 'Grape__Healthy', 'Grape__Rotten',
                'Guava__Healthy', 'Guava__Rotten', 'Jujube__Healthy', 'Jujube__Rotten',
                'Mango__Healthy', 'Mango__Rotten', 'Pomegranate__Healthy', 'Pomegranate__Rotten',
                'Strawberry__Healthy', 'Strawberry__Rotten', 'Carrot__Healthy', 'Carrot__Rotten']

# ✅ Streamlit config
st.set_page_config(page_title="Fruit & Veg Classifier", layout="centered")
st.title("🍎 Rotten or Fresh? Fruit & Vegetable Classifier")

# ✅ Upload images
uploaded_files = st.file_uploader("📤 Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# ✅ Camera support
camera_image = st.camera_input("📸 Or take a picture")

# ✅ Combine uploaded + camera
all_inputs = uploaded_files if uploaded_files else []
if camera_image is not None:
    all_inputs.append(camera_image)

# ✅ Prediction block
if all_inputs:
    for uploaded_file in all_inputs:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📷 Uploaded Image", use_column_width=True)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch)
        class_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        st.markdown(f"### 🔍 Predicted: `{class_labels[class_idx]}`")
        st.markdown(f"🧠 Confidence: **{confidence:.2f}%**")
        st.markdown("---")
