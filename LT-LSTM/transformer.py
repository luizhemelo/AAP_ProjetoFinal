import tensorflow
from tensorflow import initializers
from tensorflow.keras import models, layers

try:
	for device in tensorflow.config.experimental.list_physical_devices("GPU"):
		tensorflow.config.experimental.set_memory_growth(device, True)
except:
	print("Failed on enabling dynamic memory allocation on GPU devices!")

class Encoder(layers.Layer):
	def __init__(self, vocabulary_size, embedding_dimensions, encoding_units, batch_size, recurrent_layer, *args, **kwargs):
		super(Encoder, self).__init__(*args, **kwargs)
		self.vocabulary_size = vocabulary_size
		self.batch_size = batch_size
		self.encoding_units = encoding_units
		self.embedding_dimensions = embedding_dimensions
		self.recurrent_layer = recurrent_layer
		self.embedding = None
		self.initial_state = None

	def build(self, input_shape):
		self.embedding = layers.Embedding(self.vocabulary_size, self.embedding_dimensions)
		
	def __call__(self, X):
		x = self.embedding(X)
		output, state = self.recurrent_layer(X)
		return output, state

class Decoder(layers.Layer):
	def __init__(self, vocabulary_size, embedding_dimensions, decoding_units, recurrent_layer, batch_size, *args, **kwargs):
		super(Decoder, self).__init__(*args, **kwargs)
		self.batch_size = batch_size
		self.decoding_units = decoding_units
		self.vocabulary_size = vocabulary_size
		self.embedding_dimensions = embedding_dimensions
		self.recurrent_layer = recurrent_layer
		self.embedding = None

		def build(self, input_shape):
			self.embedding = layers.Embedding(self.vocabulary_size, self.embedding_dimensions)

		def __call__(self, X, H):
			return None
