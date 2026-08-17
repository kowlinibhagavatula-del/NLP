import joblib
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
# LOAD TOKENIZER
# ============================================================

tokenizer = joblib.load(
    MODEL_DIR / "tokenizer.pkl"
)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    MODEL_DIR / "label_encoder.pkl"
)


# ============================================================
# AVAILABLE MODELS
# ============================================================

models = {

    "1": (
        "SimpleRNN",
        MODEL_DIR / "simple_rnn_model.keras"
    ),

    "2": (
        "LSTM",
        MODEL_DIR / "lstm_model.keras"
    ),

    "3": (
        "GRU",
        MODEL_DIR / "gru_model.keras"
    ),

    "4": (
        "Transformer",
        MODEL_DIR / "transformer_model.keras"
    )
}


# ============================================================
# DISPLAY MENU
# ============================================================

print("=" * 60)
print("LANGUAGE DETECTION SYSTEM")
print("=" * 60)

print()
print("Choose Model")
print("1. SimpleRNN")
print("2. LSTM")
print("3. GRU")
print("4. Transformer")


choice = input(
    "\nEnter choice (1-4): "
).strip()


if choice not in models:

    print("Invalid choice.")

    raise SystemExit


model_name, model_path = models[choice]


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(f"Loading {model_name} Model...")


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


print(
    f"{model_name} Model Loaded Successfully!"
)

print()
print("Type 'exit' to stop.")


# ============================================================
# PREDICTION LOOP
# ============================================================

while True:

    sentence = input(
        "\nEnter Text : "
    ).strip()


    if sentence.lower() == "exit":

        print("Program ended.")

        break


    if not sentence:

        print("Please enter some text.")

        continue


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    sequence = tokenizer.texts_to_sequences(
        [sentence]
    )


    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    padded = tf.keras.utils.pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        padded,
        verbose=0
    )


    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_index = int(
        prediction[0].argmax()
    )


    language = label_encoder.inverse_transform(
        [predicted_index]
    )[0]


    confidence = (
        prediction[0][predicted_index] * 100
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("-" * 40)

    print(
        "Model      :",
        model_name
    )

    print(
        "Language   :",
        language
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    print("-" * 40)