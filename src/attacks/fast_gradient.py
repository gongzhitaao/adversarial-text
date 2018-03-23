import tensorflow as tf


__all__ = ['fgm']


def fgm(model, x, eps=0.01, epochs=1, sign=True, clip_min=0., clip_max=1.):
    ybar = model(x)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar = model.predict_from_embedding(xadv)
        loss = loss_fn(labels=target, logits=model.logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    zadv = model.embed(x)
    zadv, _ = tf.while_loop(_cond, _body, (zadv, 0), back_prop=False,
                            name='fast_gradient')
    return zadv
