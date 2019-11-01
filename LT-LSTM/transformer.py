import tensorflow
from tensorflow.keras import models, layers

try:
	for device in tensorflow.config.experimental.list_physical_devices("GPU"):
		tensorflow.config.experimental.set_memory_growth(device, True)
except:
	print("Failed on enabling dynamic memory allocation on GPU devices!")

class Encoder(models.Model):
	def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
		super(Encoder, self).__init__()
		self.batch_sz = batch_sz
		self.enc_units = enc_units
		self.embedding = layers.Embedding(vocab_size, embedding_dim)
		self.gru = layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer=initializers.GlorotUniform())

	def call(self, x, hidden):
		x = self.embedding(x)
		output, state = self.gru(x, initial_state=hidden)
		return output, state

	def initialize_hidden_state(self):
		return tensorflow.zeros((self.batch_sz, self.enc_units))

class Decoder(models.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
		super(Decoder, self).__init__()
		self.batch_sz = batch_sz
		self.dec_units = dec_units
		self.embedding = layers.Embedding(vocab_size, embedding_dim)
		self.gru = layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer=initializers.GlorotUniform())
		self.fc = layers.Dense(vocab_size)
		# used for attention
		self.attention = BahdanauAttention(self.dec_units)

	def call(self, x, hidden, enc_output):
		# enc_output shape == (batch_size, max_length, hidden_size)
		context_vector, attention_weights = self.attention(hidden, enc_output)
		# x shape after passing through embedding == (batch_size, 1, embedding_dim)
		x = self.embedding(x)
		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		x = tensorflow.concat([tensorflow.expand_dims(context_vector, 1), x], axis=-1)
		# passing the concatenated vector to the GRU
		output, state = self.gru(x)
		# output shape == (batch_size * 1, hidden_size)
		output = tensorflow.reshape(output, (-1, output.shape[2]))
		# output shape == (batch_size, vocab)
		x = self.fc(output)
		return x, state, attention_weights

class BahdanauAttention(layers.Layer):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = layers.Dense(units)
		self.W2 = layers.Dense(units)
		self.V = layers.Dense(1)

	def call(self, query, values):
		# hidden shape == (batch_size, hidden size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden size)
		# we are doing this to perform addition to calculate the score
		hidden_with_time_axis = tensorflow.expand_dims(query, 1)
		# score shape == (batch_size, max_length, 1)
		# we get 1 at the last axis because we are applying score to self.V
		# the shape of the tensor before applying self.V is (batch_size, max_length, units)
		score = self.V(tensorflow.math.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
		# attention_weights shape == (batch_size, max_length, 1)
		attention_weights = tensorflow.math.softmax(score, axis=1)
		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * values
		context_vector = tensorflow.reduce_sum(context_vector, axis=1)
		return context_vector, attention_weights    