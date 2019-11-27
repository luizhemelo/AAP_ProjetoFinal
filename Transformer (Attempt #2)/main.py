from model_bidirectional import get_model

import os
import yaml
import numpy
import logging
import tensorflow
from sklearn import model_selection
from tensorflow.keras import layers, models, preprocessing, utils, optimizers, losses, Input

try:
	device = tensorflow.config.experimental.list_physical_devices("GPU")[0]
except:
	print("No GPU avaliable!")
else:
	try:
		tensorflow.config.experimental.set_memory_growth(device, True)
	except:
		print("Could not enable dynamic memory growth to device " + str(device))

def get_logger(mod_name, log_dir):
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)

	config_filepath = os.path.join(os.path.realpath(os.path.dirname(__file__)), "logger_config.yml")
	if os.path.exists(config_filepath):
		with open(config_filepath, 'r') as f:
			config = yaml.safe_load(f.read())
			config["handlers"]["file"]["filename"] = os.path.join(log_dir, mod_name + ".log")
			logging.config.dictConfig(config)
	else:
		logging.basicConfig(level=logging.INFO)

	logger = logging.getLogger(mod_name)
	logger.info("Started log {}".format(os.path.join(log_dir, mod_name)))
	return logger

def read_data(filename):
	""" Reading the zip file to extract text """
	text = []
	with open(filename, 'r', encoding="utf-8") as f:
		i = 0
		for row in f:
			text.append(row)
			i += 1
	return text

def sents2sequences(tokenizer, sentences, reverse=False, pad_length=None, padding_type='post'):
	encoded_text = tokenizer.texts_to_sequences(sentences)
	preproc_text = preprocessing.sequence.pad_sequences(encoded_text, padding=padding_type, maxlen=pad_length)
	if reverse:
		preproc_text = numpy.flip(preproc_text, axis=1)

	return preproc_text

def preprocess_data(en_tokenizer, fr_tokenizer, en_text, fr_text, en_timesteps, fr_timesteps):
	""" Preprocessing data and getting a sequence of word indices """

	en_seq = sents2sequences(en_tokenizer, en_text, reverse=False, padding_type='pre', pad_length=en_timesteps)
	fr_seq = sents2sequences(fr_tokenizer, fr_text, pad_length=fr_timesteps)
	logger.info('Vocabulary size (English): {}'.format(numpy.max(en_seq)+1))
	logger.info('Vocabulary size (French): {}'.format(numpy.max(fr_seq)+1))
	logger.debug('En text shape: {}'.format(en_seq.shape))
	logger.debug('Fr text shape: {}'.format(fr_seq.shape))

	return en_seq, fr_seq

def get_data(logger, project_path, train_size, random_seed=100):

	""" Getting randomly shuffled training / testing data """
	en_text = read_data(os.path.join(project_path, 'data', 'small_vocab_en.txt'))
	fr_text = read_data(os.path.join(project_path, 'data', 'small_vocab_fr.txt'))
	logger.info('Length of text: {}'.format(len(en_text)))

	fr_text = ['sos ' + sent[:-1] + 'eos' if sent.endswith('.') else 'sos ' + sent + ' eos' for sent in fr_text]

	numpy.random.seed(random_seed)
	inds = numpy.arange(len(en_text))
	numpy.random.shuffle(inds)

	train_inds = inds[:train_size]
	test_inds = inds[train_size:]
	tr_en_text = [en_text[ti] for ti in train_inds]
	tr_fr_text = [fr_text[ti] for ti in train_inds]

	ts_en_text = [en_text[ti] for ti in test_inds]
	ts_fr_text = [fr_text[ti] for ti in test_inds]

	logger.info("Average length of an English sentence: {}".format(
		numpy.mean([len(en_sent.split(" ")) for en_sent in tr_en_text])))
	logger.info("Average length of a French sentence: {}".format(
		numpy.mean([len(fr_sent.split(" ")) for fr_sent in tr_fr_text])))
	return tr_en_text, tr_fr_text, ts_en_text, ts_fr_text

def infer_nmt(encoder_model, decoder_model, test_en_seq, en_vsize, fr_vsize, fr_tokenizer, fr_index2word):
	"""
	Infer logic
	:param encoder_model: keras.Model
	:param decoder_model: keras.Model
	:param test_en_seq: sequence of word ids
	:param en_vsize: int
	:param fr_vsize: int
	:return:
	"""

	test_fr_seq = sents2sequences(fr_tokenizer, ['sos'], fr_vsize)
	test_en_onehot_seq = utils.to_categorical(test_en_seq, num_classes=en_vsize)
	test_fr_onehot_seq = numpy.expand_dims(utils.to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

	enc_outs, enc_fwd_state, enc_back_state = encoder_model.predict(test_en_onehot_seq)
	dec_state = numpy.concatenate([enc_fwd_state, enc_back_state], axis=-1)
	attention_weights = []
	fr_text = ''

	for i in range(fr_timesteps):

		dec_out, attention, dec_state = decoder_model.predict(
			[enc_outs, dec_state, test_fr_onehot_seq])
		dec_ind = numpy.argmax(dec_out, axis=-1)[0, 0]

		if dec_ind == 0:
			break
		test_fr_seq = sents2sequences(fr_tokenizer, [fr_index2word[dec_ind]], fr_vsize)
		test_fr_onehot_seq = numpy.expand_dims(utils.to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

		attention_weights.append((dec_ind, attention))
		fr_text += fr_index2word[dec_ind] + ' '

	return fr_text, attention_weights

hidden_size = 1024
batch_size = None
en_timesteps = 20
en_vsize = 30
fr_timesteps = 20
fr_vsize = 20

train_size = 10000
base_dir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
logger = get_logger("examples.nmt_bidirectional.train", os.path.join(base_dir, 'logs'))
project_path = os.path.dirname(os.path.abspath(__file__))
tr_en_text, tr_fr_text, ts_en_text, ts_fr_text = get_data(logger, project_path, train_size)

en_tokenizer = preprocessing.text.Tokenizer(oov_token='UNK')
en_tokenizer.fit_on_texts(tr_en_text)

fr_tokenizer = preprocessing.text.Tokenizer(oov_token='UNK')
fr_tokenizer.fit_on_texts(tr_fr_text)

en_seq, fr_seq = preprocess_data(en_tokenizer, fr_tokenizer, tr_en_text, tr_fr_text, en_timesteps, fr_timesteps)
en_vsize = max(en_tokenizer.index_word.keys()) + 1
fr_vsize = max(fr_tokenizer.index_word.keys()) + 1

model, encoder, decoder = get_model(hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize)
model.compile(optimizer=optimizers.Adam(), loss=losses.CategoricalCrossentropy())
model.summary()

en_onehot_seq = utils.to_categorical(en_seq, num_classes=en_vsize)
fr_onehot_seq = utils.to_categorical(fr_seq, num_classes=fr_vsize)

x_train = [en_onehot_seq, fr_onehot_seq]
y_train = fr_onehot_seq

n_epochs = 10
history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=64, validation_data=(x_train, y_train))

en_tokenizer = preprocessing.text.Tokenizer(oov_token='UNK')
en_tokenizer.fit_on_texts(ts_en_text)

fr_tokenizer = preprocessing.text.Tokenizer(oov_token='UNK')
fr_tokenizer.fit_on_texts(ts_fr_text)

fr_index2word = dict(zip(fr_tokenizer.word_index.values(), fr_tokenizer.word_index.keys()))

ts_en_seq, ts_fr_seq = preprocess_data(en_tokenizer, fr_tokenizer, ts_en_text, ts_fr_text, en_timesteps, fr_timesteps)
en_vsize = max(en_tokenizer.index_word.keys()) + 1
fr_vsize = max(fr_tokenizer.index_word.keys()) + 1

text, attention_weights = infer_nmt(encoder, decoder, ts_en_seq, en_vsize, fr_vsize, fr_tokenizer, fr_index2word)
print(ts_fr_text)
print(text)