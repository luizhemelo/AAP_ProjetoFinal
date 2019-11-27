import prunable_layers
from attention import AttentionLayer

import tensorflow
from tensorflow.keras import layers, models, Input


try:
	device = tensorflow.config.experimental.list_physical_devices("GPU")[0]
except:
	print("No GPU avaliable!")
else:
	try:
		tensorflow.config.experimental.set_memory_growth(device, True)
	except:
		print("Could not enable dynamic memory growth to device " + str(device))

def get_model(hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize):
	# Define an input sequence and process it.
	if batch_size:
		encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name="encoder_inputs")
		decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps, fr_vsize), name="decoder_inputs")
	else:
		encoder_inputs = Input(shape=(en_timesteps, en_vsize), name="encoder_inputs")
		decoder_inputs = Input(shape=(fr_timesteps, fr_vsize), name="decoder_inputs")

	# Encoder GRU
	encoder_gru = layers.Bidirectional(prunable_layers.PrunableGRU(hidden_size, return_sequences=True, return_state=True, name="encoder_gru"), name="bidirectional_encoder")
	encoder_out, encoder_fwd_state, encoder_back_state = encoder_gru(encoder_inputs)

	# Set up the decoder GRU, using `encoder_states` as initial state.
	decoder_gru = prunable_layers.PrunableGRU(hidden_size * 2, return_sequences=True, return_state=True, name="decoder_gru")
	decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=layers.Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state]))

	# Attention layer
	attn_layer = AttentionLayer(name='attention_layer')
	attn_out, attn_states = attn_layer([encoder_out, decoder_out])

	# Concat attention input and decoder GRU output
	decoder_concat_input = layers.Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

	# Dense layer
	dense = prunable_layers.PrunableDense(fr_vsize, activation='softmax', name='softmax_layer')
	dense_time = layers.TimeDistributed(dense, name='time_distributed_layer')
	decoder_pred = dense_time(decoder_concat_input)

	# Full model
	full_model = models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)

	""" Inference model """
	batch_size = 1

	""" Encoder (Inference) model """
	encoder_inf_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inf_inputs')
	encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state = encoder_gru(encoder_inf_inputs)
	encoder_model = models.Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state])

	""" Decoder (Inference) model """
	decoder_inf_inputs = Input(batch_shape=(batch_size, 1, fr_vsize), name='decoder_word_inputs')
	encoder_inf_states = Input(batch_shape=(batch_size, en_timesteps, 2*hidden_size), name='encoder_inf_states')
	decoder_init_state = Input(batch_shape=(batch_size, 2*hidden_size), name='decoder_init')

	decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
	attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
	decoder_inf_concat = layers.Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
	decoder_inf_pred = layers.TimeDistributed(dense)(decoder_inf_concat)
	decoder_model = models.Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs], outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

	return full_model, encoder_model, decoder_model