
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Glaucoma Screening",
    page_icon="👁️"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "glaucoma_efficientnet_model.keras"
    )

model = load_model()
base_model = model.layers[0]

# Find last convolutional layer
conv_layers = [
    layer for layer in base_model.layers
    if isinstance(layer, tf.keras.layers.Conv2D)
]

last_conv_layer = conv_layers[-1]


def predict_image(image):

    image = image.convert("RGB")
    img = image.resize((224, 224))

    img_array = np.array(img).astype("float32")
    input_tensor = tf.expand_dims(img_array, axis=0)

    # Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_output, features = grad_model(input_tensor)

        x = model.layers[1](features)
        x = model.layers[2](x)
        prediction = model.layers[3](x)

    gradients = tape.gradient(
        prediction,
        conv_output
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224),
        method="bilinear"
    )

    heatmap = heatmap.numpy().squeeze()

    # Model output = non-glaucoma probability
    glaucoma_risk = 1 - float(prediction[0][0])

    return glaucoma_risk, heatmap, np.array(img)


# -------------------------
# WEBSITE
# -------------------------

st.title("👁️ AI-Based Glaucoma Risk Screening")

st.write(
    "Upload a retinal fundus image for AI-based glaucoma "
    "risk screening and Grad-CAM visualization."
)

uploaded_file = st.file_uploader(
    "Upload Fundus Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Fundus Image",
        use_container_width=True
    )

    if st.button("🔍 Analyze Image"):

        with st.spinner("AI is analyzing the image..."):

            risk, heatmap, processed_image = predict_image(image)

        st.subheader("AI Screening Result")

        st.metric(
            "Glaucoma Risk",
            f"{risk * 100:.1f}%"
        )

        if risk >= 0.50:

            st.error(
                "Glaucoma Suspected"
            )

            st.write(
                "Further ophthalmic evaluation is recommended."
            )

        else:

            st.success(
                "Lower Estimated Glaucoma Risk"
            )

        st.subheader(
            "AI Explainability — Grad-CAM"
        )

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.imshow(processed_image)

        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            interpolation="bilinear"
        )

        ax.axis("off")

        st.pyplot(
            fig,
            clear_figure=True
        )

        st.warning(
            "Research prototype only. This AI result is not "
            "a medical diagnosis and does not directly measure "
            "intraocular pressure (IOP)."
        )
