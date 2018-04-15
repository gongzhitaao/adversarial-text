"""
Implementation of Char-level LSTM for text classification
"""
import tensorflow as tf

from highway import Highway


class CharLSTM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.build = False

    def _build(self):
        cfg = self.cfg
        if cfg.embedding is None:
            self.embedding = tf.get_variable(
                'embedding', [cfg.vocab_size, cfg.embedding_dim])
        else:
            self.embedding = tf.get_variable(
                name='embedding', initializer=cfg.embedding, trainable=False)
        self.conv1ds = [tf.layers.Conv1D(filters, kernel_size, use_bias=True,
                                         activation=tf.tanh)
                        for filters, kernel_size in zip(cfg.feature_maps,
                                                        cfg.kernel_sizes)]
        self.highways = [Highway() for _ in range(cfg.highways)]
        self.cells = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.LSTMCell(cfg.lstm_units)
            for _ in range(cfg.lstms)])
        self.dropout = tf.layers.Dropout(rate=cfg.drop_rate)
        if 2 == cfg.n_classes:
            self.resize = tf.layers.Dense(1)
        else:
            self.resize = tf.layers.Dense(cfg.n_classes)

        self.x_embed = None
        self.logits = None
        self.build = True

    def _inference_from_embedding(self, z, training):
        z = self.dropout(z, training=training)
        zs = [conv1d(z) for conv1d in self.conv1ds]
        zs = [tf.reduce_max(z, axis=1) for z in zs]  # max-over-time
        z = tf.concat(zs, axis=1)
        z = tf.expand_dims(z, axis=1)
        z, _ = tf.nn.dynamic_rnn(self.cells, z, dtype=tf.float32)
        z = tf.squeeze(z)
        z = self.dropout(z, training=training)
        self.logits = z = self.resize(z)
        return z

    def embed(self, x):
        if not self.build:
            self._build()
        self.x_embed = tf.nn.embedding_lookup(self.embedding, x)
        return self.x_embed

    def predict(self, x, training=False):
        self.x_embed = self.embed(x)
        logits = self._inference_from_embedding(self.x_embed, training)
        y = self.cfg.output(logits)
        return y

    def __call__(self, x, training=False):
        return self.predict(x, training)

    def predict_from_embedding(self, x_embed, training=False):
        if not self.build:
            self._build()
        logits = self._inference_from_embedding(x_embed, training)
        y = self.cfg.output(logits)
        return y

    def reverse_embedding(self, x_embed):
        if not self.build:
            self._build()
        # [B, L, V] = [B, L, D] x [D, V], actually D = V in our case.
        # https://stackoverflow.com/a/46568057/1429714
        dist = tf.tensordot(tf.nn.l2_normalize(x_embed, axis=-1),
                            tf.transpose(self.embedding), axes=1)
        # [B, L]
        token_ids = tf.argmax(dist, axis=-1)
        return token_ids
