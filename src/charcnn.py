"""
Implementation of Char-level CNN for text classification
"""
import tensorflow as tf

from highway import Highway


class CharCNN:
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

    def _add_inference_graph(self, x, training=False):
        self.x_embed = z = tf.nn.embedding_lookup(self.embedding, x)
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

    def predict(self, x, training=False):
        if not self.build:
            self._build()
        logits = self._add_inference_graph(x, training)
        y = self.cfg.output(logits)
        return y

    def __call__(self, x, training=False):
        return self.predict(x, training)
