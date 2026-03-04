import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = "mnist_cnn.keras"
CLASS_NAMES = [str(i) for i in range(10)]  # "0"..."9"

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def preprocess_mnist(pil_img: Image.Image) -> np.ndarray:
    """
    Converts an uploaded image into MNIST-like format:
    - grayscale
    - invert if background looks white
    - center/resize to 28x28
    - normalize to [0,1]
    - shape = (1, 28, 28, 1)
    """
    # Convert to grayscale
    img = pil_img.convert("L")

    # Optional: make background closer to MNIST style (white digit on black background)
    # Many uploaded images are black digit on white background.
    # We'll auto-invert if the average pixel is bright (white-ish background).
    np_img = np.array(img)
    if np.mean(np_img) > 127:
        img = ImageOps.invert(img)

    # Resize to 28x28 (simple approach)
    img = img.resize((28, 28))

    # Normalize
    arr = np.array(img).astype("float32") / 255.0  # (28,28)

    # Add channel + batch dims
    arr = arr[None, ..., None]  # (1,28,28,1)
    return arr

def predict(arr: np.ndarray):
    probs = model.predict(arr, verbose=0)[0]  # (10,)
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

st.title("✍️ MNIST Digit Classifier (0–9)")
st.write("Upload an image of a single digit and get the predicted class + probability distribution.")

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(uploaded)

    st.image(pil_img, caption="Uploaded image", use_container_width=True)

    x = preprocess_mnist(pil_img)

    # Show the processed 28x28 image (debug/UX)
    st.subheader("Processed (28×28) image fed to the model")
    st.image(x[0, :, :, 0], clamp=True, use_container_width=False)

    with st.spinner("Predicting..."):
        pred_idx, probs = predict(x)

    st.subheader("Result")
    st.metric("Predicted digit", CLASS_NAMES[pred_idx], f"{probs[pred_idx]*100:.2f}%")

    st.write("Probabilities (softmax):")
    prob_table = {str(i): float(probs[i]) for i in range(10)}
    st.bar_chart(prob_table)
else:
    st.info("Upload an image to start.")
