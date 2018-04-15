import tensorflow as tf


__all__ = ['hf_replace']


def hf_replace(model, x, embedding_dim, seqlen,
               loss_fn=tf.nn.sigmoid_cross_entropy_with_logits, beam_width=1,
               chars=10):
    """Hotflip replace attack.

    See <https://arxiv.org/abs/1712.06751> for details.  Only the character
    level replace attack is implemented.  Note that this attack is
    specifically designed for character-level text classification model
    adapted from <https://arxiv.org/abs/1508.06615>.  And the implementation
    actually depends upon the architecture.
    """
    D = embedding_dim              # char embedding dimension
    L = seqlen                     # number of chars in a sequence
    W = beam_width                 # beam size
    N = chars                      # maximum number of chars to change
    B = x.get_shape().as_list()[0]  # batch size
    y = tf.stop_gradient(model(x))

    def _beam_step(x):
        # x: [B, L]
        model.predict(x)
        z = model.x_embed
        loss = loss_fn(labels=y, logits=model.logits)
        grad = tf.gradients(loss, z)[0]
        # [B, L, 1]
        target_grad = tf.reduce_max(grad*z, axis=-1, keepdims=True)
        multiples = tf.one_hot(tf.rank(z)-1, tf.rank(z), off_value=1,
                               on_value=z.get_shape().as_list()[-1])
        # [B, L, D]
        target_grad = tf.tile(target_grad, multiples)
        # [B, L, D]
        score = grad - target_grad
        # [B, LD]
        score = tf.reshape(score, [B, -1])
        # [B, W]
        vals, inds = tf.nn.top_k(score, k=W)
        # W [B]
        char_indices = tf.unstack(inds/D, axis=1)
        # W [B, L, 1]
        masks = [tf.expand_dims(tf.one_hot(tf.to_int32(ind), L, on_value=0.0,
                                           off_value=1.0), axis=-1)
                 for ind in char_indices]
        # W [B, L, D]
        dzs = [tf.reshape(tf.one_hot(ind, L*D, on_value=1.0, off_value=0.0),
                          [B, L, D])
               for ind in tf.unstack(inds, axis=1)]
        # W [B, L, D]
        zadv = [z*mask+dz for mask, dz in zip(masks, dzs)]
        # [W, B, L, D]
        zadv = tf.stack(zadv)
        # [B, W, L, D]
        zadv = tf.transpose(zadv, [1, 0, 2, 3])
        # [B, W, L]
        xadv = tf.argmax(zadv, axis=-1, output_type=tf.int32)
        return xadv, vals

    def _cond(i, *_):
        return tf.less(i, N)

    def _body(i, x_beam):
        # ([W, B, W, L], [W, B, W])
        x_advs, scores = tf.map_fn(_beam_step, x_beam, back_prop=False,
                                   dtype=(tf.int32, tf.float32))
        # [B, WW, L]
        x_adv = tf.concat(tf.unstack(x_advs), axis=1)
        # [B, WW]
        score = tf.concat(tf.unstack(scores), axis=1)
        # [B, W]
        vals, inds = tf.nn.top_k(score, k=W)
        # [BW]
        a1 = tf.reshape(inds, [-1])
        multiples = tf.one_hot(1, 2, off_value=1, on_value=W)
        # [B, W]
        a0 = tf.tile(tf.reshape(tf.range(B), [-1, 1]), multiples)
        # [BW]
        a0 = tf.reshape(a0, [-1])
        # [BW, 2]
        ind = tf.stack((a0, a1), axis=1)
        # [BW, L]
        x_new_beam = tf.gather_nd(x_adv, ind)
        # [B, W, L]
        x_new_beam = tf.reshape(x_new_beam, [B, W, L])
        # [W, B, L]
        x_new_beam = tf.transpose(x_new_beam, [1, 0, 2])
        return i+1, x_new_beam

    # [1, B, L]
    xadv = tf.expand_dims(x, axis=0)
    # [W, B, L]
    xadv = tf.tile(xadv, [W, 1, 1])
    _, xadv = tf.while_loop(_cond, _body, (0, xadv), back_prop=False)
    return xadv
