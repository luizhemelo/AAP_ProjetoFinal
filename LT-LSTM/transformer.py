import tensorflow
import prunable_layers
from tensorflow import initializers
from tensorflow.keras import models, layers

try:
	for device in tensorflow.config.experimental.list_physical_devices("GPU"):
		tensorflow.config.experimental.set_memory_growth(device, True)
except:
	print("Failed on enabling dynamic memory allocation on GPU devices!")

class BahdanauAttention(layers.Layer):
	def __init__(self, units, **kwargs):
		super(BahdanauAttention, self).__init__(**kwargs)
		self.W1 = prunable_layers.PrunableDense(units)
		self.W2 = prunable_layers.PrunableDense(units)
		self.V = prunable_layers.PrunableDense(1)

	def __call__(self, query, values):
		H_with_time_axis = tensorflow.expand_dims(query, 1)
		score = self.V(tensorflow.math.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
		attention_W = tensorflow.math.softmax(score, axis=1)
		context_tensor = tensorflow.math.reduce_sum(attention_W * values, axis=1)
		return context_tensor, attention_W


class Encoder(layers.Layer):
	def __init__(self, vocabulary_size, embedding_dimensions, encoding_units, batch_size, recurrent_layer, **kwargs):
		super(Encoder, self).__init__(**kwargs)
		self.vocabulary_size = vocabulary_size
		self.batch_size = batch_size
		self.encoding_units = encoding_units
		self.embedding_dimensions = embedding_dimensions
		self.recurrent_layer = recurrent_layer
		self.embedding = None
		self.initial_state = None

	def build(self, input_shape):
		self.embedding = layers.Embedding(self.vocabulary_size, self.embedding_dimensions)
		
	def __call__(self, X, H):
		output, state = self.recurrent_layer(X, initial_state=H)
		return output, state, H

class Decoder(layers.Layer):
	def __init__(self, vocabulary_size, embedding_dimensions, decoding_units, recurrent_layer, batch_size, **kwargs):
		super(Decoder, self).__init__(**kwargs)
		self.batch_size = batch_size
		self.decoding_units = decoding_units
		self.vocabulary_size = vocabulary_size
		self.embedding_dimensions = embedding_dimensions
		self.recurrent_layer = recurrent_layer
		self.embedding = None
		self.attention = None

		def build(self, input_shape):
			self.embedding = layers.Embedding(self.vocabulary_size, self.embedding_dimensions)
			self.attention = BahdanauAttention(self.decoding_units)

		def __call__(self, X, E, H):
			context_tensor, attention = self.attention(H, E)
			x = self.embedding(X)
			c = tensorflow.concat([tensorflow.expend_dims(context_tensor, 1), x], axis=-1)
			output, state = self.recurrent_layer(c)
			output = tensorflow.reshape(output, (-1, output.shape[2]))
			return c, state, attention
