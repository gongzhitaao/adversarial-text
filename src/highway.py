"""
Tensorflow Implementation of Highway Layers.
"""
import tensorflow as tf


class Highway(tf.layers.Layer):
    """Highway layer class.

    Modified from tf.layers.Dense class.
    """
    def __init__(self,
                 units=None,
                 couple=True,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.initializers.zeros(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 transform_kernel_initializer=None,
                 transform_bias_initializer=tf.initializers.truncated_normal(
                     mean=-2.0, stddev=1.0),
                 transform_kernel_regularizer=None,
                 transform_bias_regularizer=None,
                 carry_kernel_initializer=None,
                 carry_bias_initializer=tf.initializers.zeros(),
                 carry_kernel_regularizer=None,
                 carry_bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Highway, self).__init__(
            trainable=trainable, name=name,
            activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.couple = couple
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.transform_kernel_initializer = transform_kernel_initializer
        self.transform_bias_initializer = transform_bias_initializer
        self.transform_kernel_regularizer = transform_kernel_regularizer
        self.transform_bias_regularizer = transform_bias_regularizer
        self.carry_kernel_initializer = carry_kernel_initializer
        self.carry_bias_initializer = carry_bias_initializer
        self.carry_kernel_regularizer = carry_kernel_regularizer
        self.carry_bias_regularizer = carry_bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = tf.layers.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        dim = input_shape[-1].value,
        self.transform = tf.layers.Dense(
            units=dim,
            name='dense',
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer)
        self.transform.build(input_shape)
        self.transform_gate = tf.layers.Dense(
            units=dim,
            name='transform_gate',
            activation=tf.sigmoid,
            use_bias=True,
            kernel_initializer=self.transform_kernel_initializer,
            bias_initializer=self.transform_bias_initializer,
            kernel_regularizer=self.transform_kernel_regularizer,
            bias_regularizer=self.transform_bias_regularizer)
        self.transform_gate.build(input_shape)
        if not self.couple:
            self.carry_gate = tf.layers.Dense(
                units=dim,
                name='carry_gate',
                activation=tf.sigmoid,
                use_bias=True,
                kernel_initializer=self.carry_kernel_initializer,
                bias_initializer=self.carry_bias_initializer,
                kernel_regularizer=self.carry_kernel_regularizer,
                bias_regularizer=self.carry_bias_regularizer)
            self.carry_gate.build(input_shape)
        if self.units is not None and self.units != dim:
            self.resize = True
            self.resize_op = tf.layers.Dense(
                units=self.units,
                name='resize',
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer)
            self.resize.build(input_shape)
        else:
            self.resize = False
        self.built = True

    def call(self, inputs):
        z = self.transform.apply(inputs)
        t = self.transform_gate.apply(inputs)
        if self.couple:
            c = 1 - t
        else:
            c = self.carry_gate.apply(inputs)
        outputs = z * t + inputs * c
        if self.resize:
            outputs = self.resize_op.apply(outputs)
        return outputs
