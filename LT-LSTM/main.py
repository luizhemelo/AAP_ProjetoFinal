import io
import os
import re
import time
import numpy
import tensorflow
import unicodedata
from matplotlib import pyplot
from matplotlib import ticker
from sklearn import model_selection
from tensorflow.keras import preprocessing, utils
from transformer_example_prunable import Encoder, Decoder, BahdanauAttention
from tensorflow import losses, optimizers, initializers, train

try:
	for device in tensorflow.config.experimental.list_physical_devices("GPU"):
		tensorflow.config.experimental.set_memory_growth(device, True)
except:
	print("Failed on enabling dynamic memory allocation on GPU devices!")

def unicode_to_ascii(s):
		return ''.join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def preprocess_sentence(w):
		w = unicode_to_ascii(w.lower().strip())
		# creating a space between a word and the punctuation following it
		# eg: "he is a boy." => "he is a boy ."
		w = re.sub(r"([?.!,¿])", r" \1 ", w)
		w = re.sub(r'[" "]+', " ", w)
		# replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
		w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
		w = w.rstrip().strip()
		w = "<start> " + w + " <end>"
		return w

en_sentence = u"Excuse me, may I borrow this book of Willian Shakespeare?"
pt_sentence = u"Olá, posso pegar emprestado esse livro de Willian Shakespeare?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(pt_sentence).encode("utf-8"))

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
	lines = io.open(path, encoding="UTF-8").read().strip().split('\n')
	word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
	return zip(*word_pairs)

path_to_zip = utils.get_file("spa-eng.zip", origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip", extract=True)

path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

en, sp = create_dataset(path_to_file,None)

print(en[-1])
print(sp[-1])

def max_length(tensor):
		return max(len(t) for t in tensor)

def tokenize(lang):
	lang_tokenizer = preprocessing.text.Tokenizer(filters='')
	lang_tokenizer.fit_on_texts(lang)
	tensor = lang_tokenizer.texts_to_sequences(lang)
	tensor = preprocessing.sequence.pad_sequences(tensor, padding="post")
	return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
		# creating cleaned input, output pairs
		targ_lang, inp_lang = create_dataset(path, num_examples)
		input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
		target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
		return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

#numero de sentencas que serao usadas
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = model_selection.train_test_split(input_tensor, target_tensor, test_size=0.33)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

def convert(lang, tensor):
	for t in tensor:
		if t != 0:
			print ("%d ----> %s" % (t, lang.index_word[t]))

print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print ()
print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tensorflow.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ("Encoder output shape: (batch size, sequence length, units) {}".format(sample_output.shape))
print ("Encoder Hidden state shape: (batch size, units) {}".format(sample_hidden.shape))

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tensorflow.random.uniform((64, 1)), sample_hidden, sample_output)

print ("Decoder output shape: (batch_size, vocab size) {}".format(sample_decoder_output.shape))

loss_object = losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

def loss_function(real, pred):
	mask = tensorflow.math.logical_not(tensorflow.math.equal(real, 0))
	loss_ = loss_object(real, pred)
	mask = tensorflow.cast(mask, dtype=loss_.dtype)
	loss_ *= mask
	return tensorflow.reduce_mean(loss_)

optimizer = optimizers.Adam()
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

#@tensorflow.function
def train_step(inp, targ, enc_hidden):
	loss = 0
	with tensorflow.GradientTape() as tape:
		enc_output, enc_hidden = encoder(inp, enc_hidden)
		dec_hidden = enc_hidden
		dec_input = tensorflow.expand_dims([targ_lang.word_index["<start>"]] * BATCH_SIZE, 1)
		# Teacher forcing - feeding the target as the next input
		for t in range(1, targ.shape[1]):
			# passing enc_output to the decoder
			predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
			loss += loss_function(targ[:, t], predictions)
			# using teacher forcing
			dec_input = tensorflow.expand_dims(targ[:, t], 1)

	batch_loss = (loss / int(targ.shape[1]))
	variables = encoder.trainable_variables + decoder.trainable_variables
	gradients = tape.gradient(loss, variables)
	optimizer.apply_gradients(zip(gradients, variables))
	return batch_loss


EPOCHS = 10

for epoch in range(EPOCHS):
	start = time.time()
	enc_hidden = encoder.initialize_hidden_state()
	total_loss = 0
	for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
		batch_loss = train_step(inp, targ, enc_hidden)
		total_loss += batch_loss

		if batch % 100 == 0:
				print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy()))
	# saving (checkpoint) the model every 2 epochs
	if (epoch + 1) % 2 == 0:
		checkpoint.save(file_prefix = checkpoint_prefix)

	print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / steps_per_epoch))
	print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

def evaluate(sentence):
		attention_plot = numpy.zeros((max_length_targ, max_length_inp))
		sentence = preprocess_sentence(sentence)
		inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
		inputs = preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding="post")
		inputs = tensorflow.convert_to_tensor(inputs)
		result = ""
		hidden = [tensorflow.zeros((1, units))]
		enc_out, enc_hidden = encoder(inputs, hidden)

		dec_hidden = enc_hidden
		dec_input = tensorflow.expand_dims([targ_lang.word_index["<start>"]], 0)

		for t in range(max_length_targ):
				predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
				# storing the attention weights to plot later on
				attention_weights = tensorflow.reshape(attention_weights, (-1,))
				attention_plot[t] = attention_weights.numpy()
				predicted_id = tensorflow.argmax(predictions[0]).numpy()
				result += targ_lang.index_word[predicted_id] + ' '
				if targ_lang.index_word[predicted_id] == "<end>":
						return result, sentence, attention_plot
				# the predicted ID is fed back into the model
				dec_input = tensorflow.expand_dims([predicted_id], 0)
		return result, sentence, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
		figure = pyplot.figure(figsize=(10,10))
		axis = figure.add_subplot(1, 1, 1)
		axis.matshow(attention, cmap="viridis")
		fontdict = {"fontsize": 14}
		axis.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
		axis.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
		axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
		axis.yaxis.set_major_locator(ticker.MultipleLocator(1))
		pyplot.show()

def translate(sentence):
		result, sentence, attention_plot = evaluate(sentence)
		print("Input: %s" % (sentence))
		print("Predicted translation: {}".format(result))
		attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
		plot_attention(attention_plot, sentence.split(' '), result.split(' '))

translate(u"hace mucho frio aqui.")
translate(u"esta es mi vida.")
translate(u"¿todavia estan en casa?")
# wrong translation
translate(u"trata de averiguarlo.")