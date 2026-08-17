import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="LanguageDetection"
)
class TokenAndPositionEmbedding(tf.keras.layers.Layer):

    def __init__(
        self,
        maxlen,
        vocab_size,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim
        )

    def call(self, inputs):

        positions = tf.range(
            start=0,
            limit=tf.shape(inputs)[-1],
            delta=1
        )

        positions = self.pos_emb(positions)

        token_embeddings = self.token_emb(inputs)

        return token_embeddings + positions

    def get_config(self):

        config = super().get_config()

        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })

        return config


@tf.keras.utils.register_keras_serializable(
    package="LanguageDetection"
)
class TransformerBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                ff_dim,
                activation="relu"
            ),
            tf.keras.layers.Dense(embed_dim)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):

        attention_output = self.att(
            inputs,
            inputs
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        out1 = self.layernorm1(
            inputs + attention_output
        )

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        return self.layernorm2(
            out1 + ffn_output
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })

        return config