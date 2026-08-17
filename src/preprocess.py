import joblib
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATA_PATH,
    MODEL_DIR,
    VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    TEST_SIZE,
    RANDOM_STATE
)

from src.utils import clean_text


def prepare_data():
    """
    Load, clean, tokenize and split the dataset.

    Returns:
        X_train
        X_test
        y_train
        y_test
        tokenizer
        label_encoder
    """

    print("Loading dataset...")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(df.head())

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    if "Text" not in df.columns or "language" not in df.columns:

        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ["Text", "language"]
        else:
            raise ValueError(
                "Dataset must contain Text and language columns."
            )

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    df = df[["Text", "language"]]

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df.dropna(inplace=True)

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df.drop_duplicates(inplace=True)

    # --------------------------------------------------------
    # Convert to string
    # --------------------------------------------------------

    df["Text"] = df["Text"].astype(str)
    df["language"] = df["language"].astype(str)

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    df["Text"] = df["Text"].apply(clean_text)

    # Remove empty text
    df = df[df["Text"].str.len() > 0]

    # --------------------------------------------------------
    # X and y
    # --------------------------------------------------------

    X_text = df["Text"].values
    y_text = df["language"].values

    # --------------------------------------------------------
    # Label Encoding
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y_text)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(X_text)

    # --------------------------------------------------------
    # Convert text to integer sequences
    # --------------------------------------------------------

    X = tokenizer.texts_to_sequences(X_text)

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    X = tf.keras.utils.pad_sequences(
        X,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --------------------------------------------------------
    # Save tokenizer
    # --------------------------------------------------------

    joblib.dump(
        tokenizer,
        MODEL_DIR / "tokenizer.pkl"
    )

    # --------------------------------------------------------
    # Save label encoder
    # --------------------------------------------------------

    joblib.dump(
        label_encoder,
        MODEL_DIR / "label_encoder.pkl"
    )

    print()
    print("Training Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)
    print("Number of Languages :", len(label_encoder.classes_))
    print()
    print("Preprocessing Completed Successfully.")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        label_encoder
    )


# ============================================================
# Run preprocessing only when directly executed
# ============================================================

if __name__ == "__main__":

    prepare_data()