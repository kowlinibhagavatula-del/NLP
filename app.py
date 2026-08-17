import joblib
import streamlit as st
import tensorflow as tf


from src.config import (
    MODEL_DIR,
    MAX_SEQUENCE_LENGTH
)

from src.transformer_layers import (
    TokenAndPositionEmbedding,
    TransformerBlock
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Language Detection",
    page_icon="🌍",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌍 Language Detection using Deep Learning"
)


st.write(
    """
This application detects the language of text using four
deep learning models:

- SimpleRNN
- LSTM
- GRU
- Transformer
"""
)


# ============================================================
# LOAD TOKENIZER + LABEL ENCODER
# ============================================================

@st.cache_resource
def load_preprocessing_objects():

    tokenizer = joblib.load(
        MODEL_DIR / "tokenizer.pkl"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    return tokenizer, label_encoder


try:

    tokenizer, label_encoder = (
        load_preprocessing_objects()
    )

except Exception as e:

    st.error(
        "Tokenizer or Label Encoder could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# MODEL FILES
# ============================================================

model_files = {

    "SimpleRNN":
        MODEL_DIR / "simple_rnn_model.keras",

    "LSTM":
        MODEL_DIR / "lstm_model.keras",

    "GRU":
        MODEL_DIR / "gru_model.keras",

    "Transformer":
        MODEL_DIR / "transformer_model.keras"
}


# ============================================================
# MODEL SELECTION
# ============================================================

selected_model = st.selectbox(
    "Select Model",
    list(model_files.keys())
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_selected_model(
    model_name,
    model_path
):

    if model_name == "Transformer":

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "TokenAndPositionEmbedding":
                    TokenAndPositionEmbedding,

                "TransformerBlock":
                    TransformerBlock
            },
            compile=False
        )

    else:

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    return model


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

try:

    model = load_selected_model(
        selected_model,
        model_files[selected_model]
    )

    st.success(
        f"{selected_model} model loaded successfully."
    )

except Exception as e:

    st.error(
        f"Could not load {selected_model} model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# TEXT INPUT
# ============================================================

text = st.text_area(
    "Enter Text",
    height=180,
    placeholder=(
        "Example: "
        "I am learning Natural Language Processing."
    )
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict Language",
    type="primary"
):

    if not text.strip():

        st.warning(
            "Please enter some text."
        )

    else:

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        sequence = tokenizer.texts_to_sequences(
            [text]
        )


        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        padded = tf.keras.utils.pad_sequences(
            sequence,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post"
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        prediction = model.predict(
            padded,
            verbose=0
        )


        # ----------------------------------------------------
        # Get prediction
        # ----------------------------------------------------

        predicted_index = int(
            prediction[0].argmax()
        )


        language = label_encoder.inverse_transform(
            [predicted_index]
        )[0]


        confidence = (
            prediction[0][predicted_index] * 100
        )


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.success(
            f"Predicted Language: {language}"
        )


        st.info(
            f"Confidence: {confidence:.2f}%"
        )


        # ----------------------------------------------------
        # Top 5 Predictions
        # ----------------------------------------------------

        st.subheader(
            "Top 5 Predictions"
        )


        probabilities = prediction[0]

        top_indices = probabilities.argsort()[-5:][::-1]


        for index in top_indices:

            language_name = (
                label_encoder.inverse_transform(
                    [int(index)]
                )[0]
            )

            probability = (
                probabilities[index] * 100
            )


            st.write(
                f"**{language_name}** : "
                f"{probability:.2f}%"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Language Detection using "
    "SimpleRNN, LSTM, GRU & Transformer"
)