import tensorflow as tf

from sklearn.metrics import accuracy_score

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

from src.transformer_layers import (
    TokenAndPositionEmbedding,
    TransformerBlock
)


print("=" * 60)
print("TRAINING TRANSFORMER MODEL")
print("=" * 60)


# ============================================================
# PREPARE DATA
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    tokenizer,
    label_encoder
) = prepare_data()


# ============================================================
# PARAMETERS
# ============================================================

num_classes = len(label_encoder.classes_)

embed_dim = EMBEDDING_DIM

num_heads = 4

ff_dim = 256


# ============================================================
# MODEL INPUT
# ============================================================

inputs = tf.keras.Input(
    shape=(MAX_SEQUENCE_LENGTH,),
    dtype="int32"
)


# ============================================================
# TOKEN + POSITION EMBEDDING
# ============================================================

x = TokenAndPositionEmbedding(
    maxlen=MAX_SEQUENCE_LENGTH,
    vocab_size=VOCAB_SIZE,
    embed_dim=embed_dim
)(inputs)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

x = TransformerBlock(
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    rate=0.1
)(x)


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

x = tf.keras.layers.GlobalAveragePooling1D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    X_train,
    y_train,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# TEST
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print()
print("Transformer Test Accuracy:",
      f"{test_accuracy * 100:.2f}%")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "transformer_model.keras"

model.save(model_path)

print()
print("Transformer Model Saved Successfully!")
print("Saved at:", model_path)