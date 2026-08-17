import tensorflow as tf

from src.config import (
    VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    MODEL_DIR
)

from src.preprocess import prepare_data


print("=" * 60)
print("TRAINING GRU MODEL")
print("=" * 60)


(
    X_train,
    X_test,
    y_train,
    y_test,
    tokenizer,
    label_encoder
) = prepare_data()


num_classes = len(label_encoder.classes_)


model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(MAX_SEQUENCE_LENGTH,)
    ),

    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    tf.keras.layers.GRU(
        128
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping]
)


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print()
print(
    "GRU Test Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


model_path = MODEL_DIR / "gru_model.keras"

model.save(model_path)

print()
print("GRU Model Saved Successfully!")