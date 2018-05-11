import tqdm
from math import sqrt, pi
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, losses, initializers, metrics, callbacks
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
K.set_image_data_format('channels_last')

def spread_loss_wrap(epoch_num, step_per_epoch, init_margin, batch_size):
	def spread_loss(y_true, y_pred):
		"""Spread loss

		:param y_true: [None, 10] in one-hot vector
		:param y_pred: [None, 10], activation for each class
		:param margin: increment from 0.2 to 0.9 during training

		:return: spread loss
		"""
		m = max(step_per_epoch*int(epoch_num/2) +init_margin, 0.9)
		# y_pred_t (?, 1) for the true label
		y_pred_t = K.reshape(tf.boolean_mask(y_pred, K.equal(y_true, 1)), (-1, 1))
		# y_pred_i (?, 4) for the non-true label
		y_pred_i = K.reshape(tf.boolean_mask(y_pred, K.equal(y_true, 0)), (-1, int(y_pred.shape[-1] -1)))
		return K.sum(K.square(K.maximum(0.0, m -(y_pred_t -y_pred_i))), axis=-1)
	return spread_loss

def routing_alg(votes, coeffs_in, bias, iters):
	num_in = int(votes.shape[-3])
	num_out = int(votes.shape[-2])
	# normalize activations over input capsules
	priors = K.ones((1, 1, 1, num_in, num_out, 1))/num_out
	coeffs_in = coeffs_in/K.sum(coeffs_in, axis=-3, keepdims=True)
	for i in range(iters):
		# M-step
		priors = coeffs_in*priors
		sum_priors = K.sum(priors, axis=-3, keepdims=True)
		priors_in = priors/sum_priors
		# mu.shape = sigma.shape = [..., 1, num_capsule, h_capsule*w_capsule]
		mu = K.sum(priors_in*votes, axis=-3, keepdims=True)
		sigma_sq = K.sum(priors_in*K.square(votes -mu), axis=-3, keepdims=True)
		if i == iters -1:
			break
		# E-step
		log_pdf = -K.sum(K.square(votes -mu)/sigma_sq +K.log(sigma_sq*2*pi +K.epsilon()), axis=-1, keepdims=True)/2
		sum_priors_out = sum_priors/K.sum(sum_priors, axis=-2, keepdims=True)
		priors = tf.nn.softmax(K.log(sum_priors_out +K.epsilon()) +log_pdf, axis=-2)
	# activations.shape=[None, self.h_image, self.w_image, num_capsule]
	activations = K.sigmoid(bias*int(votes.shape[-1]) -sum_priors_out*K.sum(1 +K.log(sigma_sq*2*pi +K.epsilon()), axis=-1, keepdims=True)/2)
	return K.squeeze(mu, axis=-3), activations

def PrimaryCap(dim_capsules):
	def layer(inputs):
		h_image, w_image = int(inputs.shape[1]), int(inputs.shape[2])
		pose_conv = layers.Conv2D(filters=dim_capsules[0]*dim_capsules[1]*dim_capsules[2], kernel_size=1, strides=1, padding='valid', \
			kernel_initializer='glorot_uniform', bias_initializer='glorot_normal', name='prim_pose_conv')(inputs)
		coeff_conv = layers.Conv2D(filters=dim_capsules[0], kernel_size=1, strides=1, padding='valid', activation='sigmoid', \
			kernel_initializer='glorot_uniform', bias_initializer='glorot_normal', name='prim_coeff_conv')(inputs)
		return layers.Concatenate(axis=-1)([pose_conv, coeff_conv])
	return layer

def CapsuleConv(dim_capsules, kernel_size, strides, padding, name):
	def layer(inputs):
		#non-trainable convolutions
		conv = layers.SeparableConv2D(filters=dim_capsules[0]*(dim_capsules[1]*dim_capsules[2] +1), \
			kernel_size=kernel_size, strides=strides, padding=padding, depth_multiplier=1, 
			depthwise_initializer='ones', pointwise_initializer=initializers.Constant(value=1/(kernel_size*kernel_size)), use_bias=False, name=name)
		conv.trainable = False
		return conv(inputs)
	return layer

class CapsuleLayer(layers.Layer):
	"""
	The capsule layer.
	:param num_capsule: number of capsules in this layer
	:param dim_capsule: dimension of the output vectors of the capsules in this layer
	:param routings: number of iterations for the routing algorithm
	"""
	def __init__(self, num_capsule_in, dim_capsules, routings, **kwargs):
		super(CapsuleLayer, self).__init__(**kwargs)
		self.input_num_capsule = num_capsule_in
		self.num_capsule = int(dim_capsules[0])
		self.h_capsule = int(dim_capsules[1])
		self.w_capsule = int(dim_capsules[2])
		self.dim_capsule = self.h_capsule*self.w_capsule +1
		self.iters = routings

	def build(self, input_shape):
		assert len(input_shape) >= 4, \
		"The input Tensor shape should be [None, h_image, w_image, ..., input_num_capsule*input_dim_capsule]"
		self.batch_size = input_shape[0]
		self.h_image, self.w_image = int(input_shape[1]), int(input_shape[2])
		self.input_dim_capsule = int(input_shape[-1]/self.input_num_capsule)
		# Transform matrix with shape [self.num_capsule, self.input_num_capsule, 
		# self.h_capsule, int(self.input_dim_capsule/self.w_capsule)]
		assert (self.input_dim_capsule -1) % self.h_capsule == 0, \
		"The input capsule pose dimensions could not be coerced to match the pose dimensions of the output."
		self.input_w_capsule = int((self.input_dim_capsule -1)/self.h_capsule)

		self.W = self.add_weight(shape=(self.h_image, self.w_image, self.input_num_capsule, self.num_capsule, \
			self.input_w_capsule, self.w_capsule), initializer=initializers.TruncatedNormal(mean=0., stddev=1.), name='W')
		self.bias = self.add_weight(shape=(1, self.num_capsule, 1), initializer='ones', name='bias')
		self.built = True

	def call(self, inputs, training=None):
		#inflate and permute input tensor (num/dim_capsule axis fipped)
		inflate = layers.Reshape((self.h_image, self.w_image, self.input_dim_capsule, self.input_num_capsule))
		conv_out = layers.Permute((1, 2, 4, 3))(inflate(inputs))

		pose_in = layers.Reshape((self.h_image, self.w_image, self.input_num_capsule, self.h_capsule, self.input_w_capsule))(conv_out[..., :-1])
		coeffs_in = K.expand_dims(K.expand_dims(conv_out[..., -1], axis=-1), axis=-1)

		pose_votes = tf.einsum('abcmjk,bcmnij->abcmnik', pose_in, self.W)
		# pose_votes.shape=[None, self.h_image, self.w_image, input_num_capsule, num_capsule, h_capsule, w_capsule]
		assert self.h_capsule == int(pose_votes.shape[-2]), self.w_capsule == int(pose_votes.shape[-1])

		votes_in = layers.Reshape((self.h_image, self.w_image, self.input_num_capsule, self.num_capsule, self.dim_capsule -1))(pose_votes)
		poses, coeffs = routing_alg(votes_in, coeffs_in, self.bias, self.iters)
		#deflate capsules for output
		deflate = layers.Reshape((self.h_image, self.w_image, -1))
		final = K.concatenate([deflate(poses), deflate(coeffs)], axis=-1)
		return final
	def compute_output_shape(self, input_shape):
		return tuple([None, self.h_image, self.w_image, self.num_capsule*self.dim_capsule])

class ClassCapsule(CapsuleLayer):
	def __init__(self, num_capsule_in, dim_capsules, routings, **kwargs):
		super(ClassCapsule, self).__init__(num_capsule_in, dim_capsules, routings, **kwargs)
	def build(self, input_shape):
		super(ClassCapsule, self).build(input_shape)
	def call(self, inputs, training=None):
		#inflate and permute input tensor (num/dim_capsule axis fipped)
		inflate = layers.Reshape((self.h_image, self.w_image, self.input_dim_capsule, self.input_num_capsule))
		conv_out = layers.Permute((1, 2, 4, 3))(inflate(inputs))
		
		pose_in = layers.Reshape((self.h_image, self.w_image, self.input_num_capsule, self.h_capsule, self.input_w_capsule))(conv_out[..., :-1])
		coeffs_in = K.expand_dims(K.expand_dims(conv_out[..., -1], axis=-1), axis=-1)

		pose_votes = tf.einsum('abcmjk,bcmnij->abcmnik', pose_in, self.W)
		# pose_votes.shape=[None, self.h_image, self.w_image, input_num_capsule, num_capsule, h_capsule, w_capsule]
		assert self.h_capsule == int(pose_votes.shape[-2]), self.w_capsule == int(pose_votes.shape[-1])

		votes_in = layers.Reshape((self.h_image, self.w_image, self.input_num_capsule, self.num_capsule, self.dim_capsule -1))(pose_votes)
		aug_votes = self.coord_add(votes_in, self.h_image, self.w_image)
		# collapsed_coeffs.shape=[None, 1, 1, self.h_image*self.w_image*input_num_capsule]
		# collapsed_votes.shape=[None, 1, 1, self.h_image*self.w_image*input_num_capsule, num_capsule, h_capsule*w_capsule]
		collapsed_coeffs = layers.Reshape((1, 1, self.h_image*self.w_image*self.input_num_capsule, 1, 1))(coeffs_in)
		collapsed_votes = layers.Reshape((1, 1, self.h_image*self.w_image*self.input_num_capsule, self.num_capsule, \
			self.h_capsule*self.w_capsule))(aug_votes)
		poses, coeffs = routing_alg(collapsed_votes, collapsed_coeffs, self.bias, self.iters)
		return tf.squeeze(coeffs, [1, 2, -3, -1])

	def coord_add(self, votes, h_image, w_image):
		"""Coordinate addition.

		:param votes: (24, 4, 4, 32, 10, 16)
		:param H, W: spaital height and width 4

		:return votes: (24, 4, 4, 32, 10, 16)
		"""
		dim_capsule = votes.shape[-1]
		offset_hh = np.reshape((np.arange(h_image, dtype='float32') +0.50)/h_image, (1, h_image, 1, 1, 1))
		offset_ww = np.reshape((np.arange(w_image, dtype='float32') +0.50)/w_image, (1, 1, w_image, 1, 1))
		offset_h = np.zeros((1, h_image, 1, 1, 1, dim_capsule))
		offset_w = np.zeros((1, 1, w_image, 1, 1, dim_capsule))
		offset_w[..., 1], offset_h[..., 0] = offset_ww, offset_hh
		return votes +offset_h +offset_w

	def compute_output_shape(self, input_shape):
		return tuple([None, self.num_capsule])

def CapsNet_EM(input_shape, num_classes, iters, cifar=False, num_caps=(8, 16, 16), caps_kernel=3):
	"""Define the Capsule Network model
	"""
	caps_b, caps_c, caps_d = num_caps
	x = layers.Input(shape=input_shape, name='in')
	batch_norm = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, name='batch_norm')(x)
	# ReLU Conv1
	# Images shape (24, 28, 28, 1) -> conv 5x5 filters, 32 output channels, strides 2 with padding, ReLU
	# nets -> (?, 14, 14, 32)
	if cifar:
		conv0 = layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='valid', \
			activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_normal', name='conv0')(batch_norm)
		conv1 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', \
			activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_normal', name='conv1')(conv0)
	else:
		downsample = layers.AveragePooling2D(pool_size=(2, 2), padding='valid', name='downsample')(batch_norm)
		conv0 = layers.Conv2D(filters=24, kernel_size=5, strides=2, padding='valid', \
			activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_normal', name='conv0')(downsample)
		conv1 = layers.Conv2D(filters=32, kernel_size=7, strides=1, padding='valid', \
			activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_normal', name='conv1')(conv0)
	# PrimaryCaps: (?, w, h, 32) -> capsule 1x1 filter, 32 output capsule, strides 1, no padding
	# (poses (?, w, h, b, 4, 4), activations (?, w, h, b))
	caps0 = PrimaryCap((caps_b, 4, 4))(conv1)
	# ConvCaps1: (poses, activations) -> conv capsule, 3x3 kernels, strides 2, no padding
	# (poses (?, w, h, c, 4, 4), activations (?, w, h, c))
	conv2 = CapsuleConv((caps_b, 4, 4), kernel_size=caps_kernel, strides=2, padding='valid', name='conv2')(caps0)
	caps1 = CapsuleLayer(caps_b, (caps_c, 4, 4), iters, name='caps1')(conv2)
	# ConvCaps2: (poses, activations) -> conv capsule, 3x3 kernels, strides 1, no padding
	# (poses (?, w, h, d, 4, 4), activations (?, w, h, d))
	conv3 = CapsuleConv((caps_c, 4, 4), kernel_size=caps_kernel, strides=1, padding='valid', name='conv3')(caps1)
	caps2 = CapsuleLayer(caps_c, (caps_d, 4, 4), iters, name='caps2')(conv3)
	# Class capsules: (poses, activations) -> 1x1 convolution, 10 output capsules
	# (poses (?, 10, 4, 4), activations (?, num_classes))
	coeffs = ClassCapsule(caps_d, (num_classes, 4, 4), iters, name='classcaps')(caps2)
	model = models.Model(inputs=x, outputs=coeffs, name='model')
	return model

def main():
	import os
	import snorbdata
	from keras.datasets import cifar10, cifar100
	# setting the hyper parameters
	args = {'epochs':50, 'batch_size':250, 'lr': 1e-3, 'decay': 0.8, 'iters': 3, 'weights': None, 'save_dir':'./results', 'dataset': 10}
	print(args)
	if not os.path.exists(args['save_dir']):
		os.makedirs(args['save_dir'])
	# load data
	# define model
	graph = tf.Graph()
	with graph.as_default():
		tf.add_check_numerics_ops()
		if args['dataset'] == 10 or args['dataset'] == 100:
			model = CapsNet_EM(input_shape=(32, 32, 3), num_classes=args['dataset'], iters=args['iters'], cifar=True, num_caps=(16, 24, 24))
		else:
			model = CapsNet_EM(input_shape=(args['dataset'], args['dataset'], 1), num_classes=5, iters=args['iters'])
		print('-'*30 + 'Summary for Model' + '-'*30)
		model.summary()
		print('-'*30 + 'Summaries Done' + '-'*30)
		if args['dataset'] == 10:
			(x_train, y_train), (x_test, y_test) = cifar10.load_data()
			y_train, y_test = np.eye(10)[np.squeeze(y_train)], np.eye(10)[np.squeeze(y_test)]
		elif args['dataset'] == 100:
			(x_train, y_train), (x_test, y_test) = cifar100.load_data()
			y_train, y_test = np.eye(100)[np.squeeze(y_train)], np.eye(100)[np.squeeze(y_test)]
		else:
			x_train, y_train, x_test, y_test = snorbdata.load_data()
		if len(x_train.shape) < 4:
			x_train = np.expand_dims(x_train, axis=-1)
		if len(x_test.shape) < 4:
			x_test = np.expand_dims(x_test, axis=-1)
		print('Done loading data')
		# init the model weights with provided one
		if args['weights'] is not None:
			model.load_weights(args['weights'])

		log = callbacks.CSVLogger(args['save_dir'] + '/log.csv')
		tb = callbacks.TensorBoard(log_dir=args['save_dir'] + '/tensorboard-logs', batch_size=args['batch_size'],
			write_graph=True, write_images=True)
		checkpoint = callbacks.ModelCheckpoint(args['save_dir'] + '/w_{epoch:02d}.h5', monitor='val_categorical_accuracy',
			save_best_only=True, save_weights_only=True, verbose=1, period=2)
		lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args['lr'] * args['decay']**epoch)
		naan = callbacks.TerminateOnNaN()
		# compile and train model
		for e in range(args['epochs']):
			model.compile(optimizer=optimizers.Nadam(lr=args['lr']), loss=spread_loss_wrap(e, 0.2, 0.1, args['batch_size']), \
				metrics=['categorical_accuracy'])
			train_gen = ImageDataGenerator().flow(x_train, y_train, batch_size=args['batch_size'])
			test_gen = ImageDataGenerator().flow(x_test, y_test, batch_size=args['batch_size'])
			model.fit_generator(train_gen, validation_data=test_gen, initial_epoch=e, epochs=e +1, verbose=1, callbacks=[log, tb, checkpoint, lr_decay, naan])
	model.save_weights(args['save_dir'] + '/model.h5')
	print('Trained model saved to \'%s' % args['save_dir'])
	return
main()