"""
Some key layers used for constructing a Capsule Network. These layers can used to construct CapsNet on other dataset, 
not just on MNIST.
*NOTE*: some functions can be implemented in multiple ways, I keep all of them. You can try them for yourself just by
uncommenting them and commenting their counterparts.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


class Longest(layers.Layer):
    """
    Compute a one-hot code for the longest vector in the inputs. This is used to mask y_pred for the decoder network in testing.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        length = K.sqrt(K.sum(K.square(inputs), axis=-1))
        return K.one_hot(indices=K.argmax(length, 1), num_classes=K.int_shape(length)[1])
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class MaskNoise(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, ...] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, ...])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        z = keras.layers.Input(shape=[8, 3, ...])  # batch_size=8, each noise tensor contains 3 capsules with dim_vector=2
        out = Mask()([x, y, z])  # out.shape=[8, 3, ...] Masked with true labels y if the shape of y fits.
        ```
    """
    def call(self, inputs, **kwargs):
        assert len(inputs) == 3
        tensor, mask, noise = inputs
        total = layers.add([tensor, noise])
        masked = tf.multiply(total, K.expand_dims(mask, -1))
        # masked.shape=[None, num_capsule, ...]
        return masked

    def compute_output_shape(self, input_shape):
            assert input_shape[0] == input_shape[2] and input_shape[1] == input_shape[0][:2]
            return input_shape[0]


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.Orthogonal(gain=4.0)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])        

def PrimaryCap(channels, dim_capsule, kernel_size, strides, padding, initializer):
    """
    Apply Conv2D `channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, channels, width, height]
    :param dim_capsule: the dim of the output vector of capsule
    :param channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    def primary(inputs):
        conv_out =  layers.Conv2D(filters=channels*dim_capsule, kernel_size=kernel_size, strides=strides, padding=padding, \
                    use_bias=False, kernel_initializer=initializer, name='prim_conv2d')(inputs)

        return  layers.Lambda(squash, name='prim_squash')( \
                layers.Permute((2, 1), name='prim_permute')( \
                layers.Reshape((dim_capsule, -1), name='prim_flatten')(conv_out)))

    return primary

def InvPrimaryCap(channels, input_size, kernel_size, strides, padding, activation, initializer):
    """
    Reinflates capsule vectors and then applies conventional DeConv2D after upsampling.
    :param inputs: 3D tensor, shape=[None, num_capsules, dim_capsule]
    :param channels: the number of deconv filters
    :return: output tensor, shape=[None, channels, kernel_size*upsampling, kernel_size*upsampling]
    """
    def invprim(inputs):
        dim_capsule = K.int_shape(inputs)[2]
        deconvsource =  layers.Reshape((-1, input_size, input_size), name='inv_inflate')( \
                        layers.Flatten(name='inv_flatten')( \
                        layers.Permute((2, 1), name='inv_permute')(inputs)))

        return  layers.Conv2DTranspose(filters=channels*dim_capsule, kernel_size=kernel_size, strides=strides, padding=padding, \
                use_bias=False, activation=activation, kernel_initializer=initializer, name='inv_conv2d')(deconvsource)

    return invprim


"""
# The following is another way to implement primary capsule layer. This is much slower.
# Apply Conv2D `n_channels` times and concatenate all capsules
def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    outputs = []
    for _ in range(n_channels):
        output = layers.Conv2D(filters=dim_capsule, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
        outputs.append(layers.Reshape([output.get_shape().as_list()[1] ** 2, dim_capsule])(output))
    outputs = layers.Concatenate(axis=1)(outputs)
    return layers.Lambda(squash)(outputs)
"""
