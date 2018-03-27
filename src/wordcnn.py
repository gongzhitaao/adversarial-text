"""
Implementation of Word-level CNN for text classification
"""
import tensorflow as tf


def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret


class WordCNN:
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
        self.dropout = tf.layers.Dropout(rate=cfg.drop_rate)
        self.conv1d = tf.layers.Conv1D(cfg.filters, cfg.kernel_size)
        self.mlp = tf.layers.Dense(cfg.units)
        if 2 == cfg.n_classes:
            self.resize = tf.layers.Dense(1)
        else:
            self.resize = tf.layers.Dense(cfg.n_classes)

        self.x_embed = None
        self.logits = None
        self.build = True
        self.normalized_embedding = None

    def _inference_from_embedding(self, z, training):
        v0 = tf.trainable_variables()
        z = self.dropout(z, training=training)
        z = self.conv1d(z)
        z = tf.maximum(z, 0.2 * z)
        z = tf.reduce_max(z, axis=1, name='global_max_pooling')
        shape = z.get_shape().as_list()[1:]
        z = tf.reshape(z, [-1, _prod(shape)])
        z = self.mlp(z)
        z = self.dropout(z, training=training)
        z = tf.maximum(z, 0.2 * z)
        self.logits = z = self.resize(z)
        v1 = tf.trainable_variables()
        self.varlist = [v for v in v1 if v not in v0]
        return z

    def predict(self, x, training=False):
        self.x_embed = self.embed(x)
        logits = self._inference_from_embedding(self.x_embed, training)
        y = self.cfg.output(logits)
        return y

    def predict_from_embedding(self, x_embed, training=False):
        if not self.build:
            self._build()
        self.x_embed = x_embed
        logits = self._inference_from_embedding(self.x_embed, training)
        y = self.cfg.output(logits)
        return y

    def embed(self, x):
        if not self.build:
            self._build()
        self.x_embed = tf.nn.embedding_lookup(self.embedding, x)
        return self.x_embed

    def reverse_embed(self, x_embed):
        if not self.build:
            self._build()
        if self.normalized_embedding is None:
            self.normalized_embedding = tf.nn.l2_normalize(self.embedding,
                                                           axis=1)
        # [B, L, V] = [B, L, D] x [D, V]
        dist = tf.tensordot(tf.nn.l2_normalize(x_embed, axis=1),
                            tf.transpose(self.normalized_embedding), axes=1)
        # [B, L]
        token_ids = tf.argmax(dist, axis=1)
        return token_ids

    def __call__(self, x, training=False):
        return self.predict(x, training)
