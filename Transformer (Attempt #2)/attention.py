import os
import tensorflow
from tensorflow.keras import backend, initializers, layers

try:
	device = tensorflow.config.experimental.list_physical_devices("GPU")[0]
except:
	print("No GPU avaliable!")
else:
	try:
		tensorflow.config.experimental.set_memory_growth(device, True)
	except:
		print("Could not enable dynamic memory growth to device " + str(device))

class AttentionLayer(layers.Layer):
	"""
	This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
	There are three sets of weights introduced W_a, U_a, and V_a
	 """

	def __init__(self, *args, **kwargs):
		super(AttentionLayer, self).__init__(*args, **kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		# Create a trainable weight variable for this layer.
		self.W_a = self.add_weight(name='W_a', shape=(input_shape[0][2], input_shape[0][2]), initializer=initializers.he_normal(), trainable=True)
		self.U_a = self.add_weight(name='U_a', shape=(input_shape[1][2], input_shape[0][2]), initializer=initializers.he_normal(), trainable=True)
		self.V_a = self.add_weight(name='V_a', shape=(input_shape[0][2], 1), initializer=initializers.he_normal(), trainable=True)
		super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

	def call(self, inputs, verbose=False):
		"""
		inputs: [encoder_output_sequence, decoder_output_sequence]
		"""
		assert type(inputs) == list
		encoder_out_seq, decoder_out_seq = inputs
		if verbose:
			print('encoder_out_seq>', encoder_out_seq.shape)
			print('decoder_out_seq>', decoder_out_seq.shape)

		@tensorflow.function
		def energy_step(inputs, states):
			""" Step function for computing energy for a single decoder state """

			assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
			assert isinstance(states, list) or isinstance(states, tuple), assert_msg

			""" Some parameters required for shaping tensors"""
			en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
			de_hidden = inputs.shape[-1]

			""" Computing S.Wa where S=[s0, s1, ..., si]"""
			# <= batch_size*en_seq_len, latent_dim
			reshaped_enc_outputs = backend.reshape(encoder_out_seq, (-1, en_hidden))
			# <= batch_size*en_seq_len, latent_dim
			W_a_dot_s = backend.reshape(backend.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
			if verbose:
				print('wa.s>',W_a_dot_s.shape)

			""" Computing hj.Ua """
			U_a_dot_h = backend.expand_dims(backend.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
			if verbose:
				print('Ua.h>',U_a_dot_h.shape)

			""" tanh(S.Wa + hj.Ua) """
			# <= batch_size*en_seq_len, latent_dim
			reshaped_Ws_plus_Uh = backend.tanh(backend.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
			if verbose:
				print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

			""" softmax(va.tanh(S.Wa + hj.Ua)) """
			# <= batch_size, en_seq_len
			e_i = backend.reshape(backend.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
			# <= batch_size, en_seq_len
			e_i = backend.softmax(e_i)

			if verbose:
				print('ei>', e_i.shape)

			return e_i, states

		@tensorflow.function
		def context_step(inputs, states):
			""" Step function for computing ci using ei """
			# <= batch_size, hidden_size
			c_i = backend.sum(encoder_out_seq * backend.expand_dims(inputs, -1), axis=1)
			if verbose:
				print('ci>', c_i.shape)
			return c_i, states

		#@tensorflow.function
		def create_inital_state(inputs, hidden_size):
			# We are not using initial states, but need to pass something to backend.rnn funciton
			fake_state = backend.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
			fake_state = backend.sum(fake_state, axis=[1, 2])  # <= (batch_size)
			fake_state = backend.expand_dims(fake_state)  # <= (batch_size, 1)
			fake_state = backend.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
			return fake_state

		fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
		fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

		""" Computing energy outputs """
		# e_outputs => (batch_size, de_seq_len, en_seq_len)
		last_out, e_outputs, _ = backend.rnn(energy_step, decoder_out_seq, [fake_state_e])

		""" Computing context vectors """
		last_out, c_outputs, _ = backend.rnn(context_step, e_outputs, [fake_state_c])

		return c_outputs, e_outputs

	#@tensorflow.function
	def compute_output_shape(self, input_shape):
		""" Outputs produced by the layer """
		return [
			tensorflow.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][2])),
			tensorflow.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
		]