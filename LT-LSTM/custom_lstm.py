import tensorflow
from tensorflow.keras import layers

try:
	for device in tensorflow.config.experimental.list_physical_devices("GPU"):
		tensorflow.config.experimental.set_memory_growth(device, True)
except:
	print("Dynamic allocation mode for GPU device failed!")

class CustomLSTMCell(layers.LSTMCell):
	def __init__(self, *args, **kwargs):
		super(CustomLSTMCell, self).__init__(*args, **kwargs)
		self._kernel1 = None
		self._kernel2 = None
		self._bias1 = None
		self._bias2 = None
		self._recurrent_kernel1 = None
		self._recurrent_kernel2 = None
		self.trainable_channels = None
		self.trainable_bias = None
		self.trainable_recurrent_channels = None
		self.saved_W = None
		self.saved_bias = None
		self.saved_recW = None

	def build(self, input_shape):
		self._kernel1 = self.add_weight("kernel1", shape=(input_shape[-1], 4 * self.units), initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, dtype=self.dtype, trainable=True)
		self._kernel2 = tensorflow.zeros((input_shape[-1], 4 * self.units), dtype=self.dtype)
		self.trainable_channels = tensorflow.ones((input_shape[-1], 4 * self.units), dtype=self.dtype)
		self._recurrent_kernel1 = self.add_weight("recurrent_kernel1", shape=(self.units, 4 * self.units), initializer=self.recurrent_initializer, regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint, dtype=self.dtype, trainable=True)
		self._recurrent_kernel2 = tensorflow.zeros((self.units, 4 * self.units), dtype=self.dtype)
		self.trainable_recurrent_channels = tensorflow.ones((self.units, 4 * self.units), dtype=self.dtype)
		if self.use_bias:
			self._bias1 = self.add_weight("bias", shape=(4 * self.units,), initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, dtype=self.dtype, trainable=True)
			self._bias2 = tensorflow.zeros((4 * self.units,), dtype=self.dtype)
			self.trainable_bias = tensorflow.ones((4 * self.units,), dtype=self.dtype)
		self.built = True
	
	@property
	def kernel(self):
		return self.trainable_channels * self._kernel1 + (1 - self.trainable_channels) * self._kernel2

	@property
	def recurrent_kernel(self):
		return self.trainable_recurrent_channels * self._recurrent_kernel1 + (1 - self.trainable_recurrent_channels) * self._recurrent_kernel2

	@property
	def bias(self):
		if not self.use_bias:
			return None
		else:
			return self.trainable_bias * self._bias1 + (1 - self.trainable_bias) * self._bias2

	def save_kernel(self):
		self.saved_W = tensorflow.identity(self.kernel)

	def save_recurrent_kernel(self):
		self.saved_recW = tensorflow.identity(self.recurrent_kernel)

	def save_bias(self):
		assert (self.use_bias)
		self.saved_bias = tensorflow.identity(self.bias)

	def restore_kernel(self):
		self._kernel1.assign(self.saved_W)

	def restore_recurrent_kernel(self):
		self._recurrent_kernel1.assign(self.saved_recW)

	def restore_bias(self):
		self._bias1.assign(self.saved_bias)

	def prune_kernel(self, to_be_pruned):
		"""
		Prune the network layer on specific weights.
		Parameters
		---------------
		to_be_pruned: NumPy Array or Tensor of shape=kernel.shape with values in {0,  1} indicating which weights
		to keep (1) and which to drop (0).
		"""
		t = tensorflow.cast(to_be_pruned, dtype=self.dtype)
		new_pruned = 1 - tensorflow.maximum((1 - t) - (1 - self.trainable_channels), 0)
		new_pruned_weights = (1 - new_pruned) * self._kernel1
		self._kernel2 += new_pruned_weights
		self.trainable_channels *= t

	def prune_recurrent_kernel(self, to_be_pruned):
		"""
		Prune the network recurrent layer on specific weights.
		Parameters
		---------------
		to_be_pruned: NumPy Array or Tensor of shape=kernel.shape with values in {0,  1} indicating which weights
		to keep (1) and which to drop (0).
		"""
		t = tensorflow.cast(to_be_pruned, dtype=self.dtype)
		new_pruned = 1 - tensorflow.maximum((1 - t) - (1 - self.trainable_recurrent_channels), 0)
		new_pruned_weights = (1 - new_pruned) * self._kernel1
		self._recurrent_kernel2 += new_pruned_weights
		self.trainable_recurrent_channels *= t

	def prune_bias(self, to_be_pruned):
		"""
		Prune the bias on specific weights.
		Parameters
		--------------
		to_be_pruned: NumPy Array or Tensor with shape=kernel.shape with values in {0,  1} indicating which weights to keep (1) and which to drop (0).
		"""
		assert (self.use_bias)
		t = tensorflow.cast(to_be_pruned, dtype=self.dtype)
		new_pruned = 1 - tensorflow.maximum((1 - t) - (1 - self.trainable_bias), 0)
		new_pruned_bias = (1 - new_pruned) * self._bias1
		self._bias2 += new_pruned_bias
		self.trainable_bias *= t

class CustomLSTM(layers.LSTM):
	def __init__(self, *args, **kwargs):
	 super(CustomLSTM, self).__init__(*args, **kwargs)
	 self.cell = CustomLSTMCell(*args, **kwargs)