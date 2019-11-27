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
		score = self.V(tensorflow.math.tanh(self.W1(values) + self.W2(H_with_time_axis)))
		attention_W = tensorflow.math.softmax(score, axis=1)
		context_tensor = tensorflow.math.reduce_sum(attention_W * values, axis=1)
		return context_tensor, attention_W


class Encoder(layers.Layer):
	def __init__(self, vocabulary_size, embedding_dimensions, encoding_units, batch_size, **kwargs):
		super(Encoder, self).__init__(**kwargs)
		self.vocabulary_size = vocabulary_size
		self.batch_size = batch_size
		self.encoding_units = encoding_units
		self.embedding_dimensions = embedding_dimensions
		self.recurrent_layer = prunable_layers.PrunableGRU(encoding_units)
		self.embedding = layers.Embedding(self.vocabulary_size, self.embedding_dimensions)
		self.initial_state = None

	def __call__(self, X, H):
		output, state = self.recurrent_layer(X, initial_state=H)
		return output, state

class Decoder(layers.Layer):
	def __init__(self, vocabulary_size, embedding_dimensions, decoding_units, batch_size, **kwargs):
		super(Decoder, self).__init__(**kwargs)
		self.batch_size = batch_size
		self.decoding_units = decoding_units
		self.vocabulary_size = vocabulary_size
		self.embedding_dimensions = embedding_dimensions
		self.recurrent_layer = prunable_layers.PrunableGRU(self.decoding_units)
		self.embedding = layers.Embedding(self.vocabulary_size, self.embedding_dimensions)
		self.attention = BahdanauAttention(self.decoding_units)

		def __call__(self, X, E, H):
			context_tensor, attention = self.attention(H, E)
			x = self.embedding(X)
			c = tensorflow.concat([tensorflow.expend_dims(context_tensor, 1), x], axis=-1)
			output, state = self.recurrent_layer(c)
			output = tensorflow.reshape(output, (-1, output.shape[2]))
			return c, state, attention

class Transformer(models.Model):
	def __init__(self, vocabulary_input_size, vocabulary_target_size, embedding_dimensions, encoding_units, decoding_units, attention_units, batch_size, target_language, **kwargs):
		super(Transformer, self).__init__(**kwargs)
		self.vocabulary_input_size = vocabulary_input_size
		self.vocabulary_target_size = vocabulary_target_size
		self.embedding_dimensions = embedding_dimensions
		self.encoding_units = encoding_units
		self.decoding_units = decoding_units
		self.attention_units = attention_units
		self.target_language = target_language
		self.batch_size = batch_size
		self.encoder = Encoder(self.vocabulary_input_size, self.embedding_dimensions, self.encoding_units, self.batch_size)
		self.decoder = Decoder(self.vocabulary_target_size, self.embedding_dimensions, self.decoding_units, self.batch_size)
		self.attention = BahdanauAttention(self.attention_units)
		self.hidden_state = tensorflow.zeros((self.batch_size, self.encoding_units))

	def __call__(self, X):
		encoding, encoding_hidden_state = self.encoder(X, self.hidden_state)
		decoding_hidden_state = encoding_hidden_state
		decoder_input = self.target_language.word_index["<start>"] * self.batch_size
		#decoder_input = tensorflow.expand_dims([] * self.batch_size, 1)
		predictions, decoding_hidden_state = self.decoder(decoder_input, encoding, decoding_hidden_state)
		self.hidden_state = decoding_hidden_state
		return predictions