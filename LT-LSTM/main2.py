# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import transformer


# %%
import io
import os
import re
import time
import numpy
import tensorflow
import unicodedata
from sklearn import model_selection
from tensorflow import losses, optimizers, initializers, metrics
from tensorflow.keras import layers, models, preprocessing, utils


# %%
try:
	for device in tensorflow.config.experimental.list_physical_devices("GPU"):
		tensorflow.config.experimental.set_memory_growth(device, True)
except:
	print("Failed on enabling dynamic memory allocation on GPU devices!")


# %%
def unicode_to_ascii(s):
		return ''.join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# %%
def preprocess_sentence(w):
	w = unicode_to_ascii(w.lower().strip())
	# creating a space between a word and the punctuation following it
	# eg: "he is a boy." => "he is a boy ."
	w = re.sub(r"([?.!,Â¿])", r" \1 ", w)
	w = re.sub(r'[" "]+', " ", w)
	# replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
	w = re.sub(r"[^a-zA-Z?.!,Â¿]+", " ", w)
	w = w.rstrip().strip()
	w = "<start> " + w + " <end>"
	return w


# %%
en_sentence = u"Excuse me, may I borrow this book of Willian Shakespeare?"
pt_sentence = u"Olá, posso pegar emprestado esse livro de Willian Shakespeare?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(pt_sentence).encode("utf-8"))


# %%
def create_dataset(path, num_examples):
	lines = io.open(path, encoding="UTF-8").read().strip().split('\n')
	word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
	return zip(*word_pairs)


# %%
path_to_zip = utils.get_file("spa-eng.zip", origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip", extract=True)
path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"
en, sp = create_dataset(path_to_file,None)
print(en[-1])
print(sp[-1])


# %%
def max_length(tensor):
	return max(len(t) for t in tensor)


# %%
def tokenize(lang):
	lang_tokenizer = preprocessing.text.Tokenizer(filters='')
	lang_tokenizer.fit_on_texts(lang)
	tensor = lang_tokenizer.texts_to_sequences(lang)
	tensor = preprocessing.sequence.pad_sequences(tensor, padding="post")
	return tensor, lang_tokenizer


# %%
def load_dataset(path, num_examples=None):
	# creating cleaned input, output pairs
	targ_lang, inp_lang = create_dataset(path, num_examples)
	input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
	target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
	return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# %%
#numero de sentencas que serao usadas
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = model_selection.train_test_split(input_tensor, target_tensor, test_size=0.33)
# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


# %%
def convert(lang, tensor):
	for t in tensor:
		if t != 0:
			print ("%d ----> %s" % (t, lang.index_word[t]))


# %%
print("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print()
print("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])


# %%
batch_size = 64
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1


# %%
t = transformer.Transformer(vocab_inp_size, vocab_tar_size, embedding_dim, units, units, 10, layers.GRU(units), batch_size, targ_lang)


# %%
def loss_function(real, pred):
	mask = tensorflow.math.logical_not(tensorflow.math.equal(real, 0))
	loss_ = loss_object(real, pred)
	mask = tensorflow.cast(mask, dtype=loss_.dtype)
	loss_ *= mask
	return tensorflow.reduce_mean(loss_)


# %%
metrics_list = [metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall(), metrics.AUC()]
t.compile(loss=loss_function, optimizer=optimizers.Adam(learning_rate=0.001), metrics=metrics_list)


# %%
t.fit(input_tensor_train, target_tensor_train, batch_size=64)


# %%
input_tensor_train.shape


# %%
target_tensor_train.shape


# %%


