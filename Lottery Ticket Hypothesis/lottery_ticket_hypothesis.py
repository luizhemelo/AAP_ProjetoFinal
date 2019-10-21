import numpy
import tensorflow
from tensorflow.keras import layers

devices = tensorflow.config.experimental.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(devices[0], True)

class PrunableDense(layers.Dense):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.trainable_channels = None
		self.trainable_bias = None
		self._kernel1 = None
		self._bias1 = None
		self._kernel2 = None
		self._bias2 = None

	def build(self, input_shape):
		last_dim = input_shape[-1]
		self._kernel1 = self.add_weight("kernel1", shape=(last_dim, self.units), initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, dtype=self.dtype, trainable=True)
		self._kernel2 = tensorflow.zeros((last_dim, self.units))
		self.trainable_channels = tensorflow.ones((last_dim, self.units))
		if self.use_bias:
			self._bias1 = self.add_weight("bias", shape=(self.units,), initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, dtype=self.dtype, trainable=True)
			self._bias2 = tensorflow.zeros((self.units,))
			self.trainable_bias = tensorflow.ones((self.units,))

	@property
	def kernel(self):
		return self.trainable_channels * self._kernel1 + (1 - self.trainable_channels * self._kernel2)

	@property
	def bias(self):
		if not self.use_bias:
			return None
		else:
			return self.trainable_bias * self._bias1 + (1 - self.trainable_bias * self._bias2)

	def prune_kernel(self, to_be_pruned):
		new_pruned = 1 - tensorflow.maximum((1 - to_be_pruned) - (1 - self.trainable_channels), 0)
		new_pruned_weights = (1 - new_pruned) * self._kernel1
		self._kernel2 += new_pruned_weights
		self.trainable_channels *= to_be_pruned

	def prune_bias(self, to_be_pruned):
		assert (self.use_bias)
		new_pruned = 1 - tensorflow.maximum((1 - to_be_pruned) - (1 - self.trainable_bias), 0)
		new_pruned_bias = (1 - new_pruned) * self._bias1
		self._bias2 += new_pruned_bias
		self.trainable_bias *= to_be_pruned
