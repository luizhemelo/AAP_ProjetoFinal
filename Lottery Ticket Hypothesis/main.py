import numpy
import tensorflow
from sklearn import preprocessing, model_selection
from lottery_ticket_hypothesis import PrunableDense
from tensorflow import optimizers, initializers, losses
from tensorflow.keras import models, layers, activations

devices = tensorflow.config.experimental.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(devices[0], True)

def create_neural_network():
	net = models.Sequential()
	net.add(layers.Dense(256, activation=activations.softsign, name="Dense0", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(layers.Dense(128, activation=activations.softsign, name="Dense1", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(layers.Dense(64, activation=activations.softsign, name="Dense2", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(layers.Dense(32, activation=activations.softsign, name="Dense3", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(layers.Dense(1, activation=activations.tanh, name="Output", kernel_initializer=initializers.he_normal(), bias_initializer=tensorflow.ones))
	return net

def create_neural_network_prunable():
	"""Prunable model of a fully-conected multilayer perceptron"""
	net = models.Sequential()
	net.add(PrunableDense(256, activation=activations.softsign, name="Dense0", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(128, activation=activations.softsign, name="Dense1", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(64, activation=activations.softsign, name="Dense2", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(32, activation=activations.softsign, name="Dense3", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(1, activation=activations.tanh, name="Output", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	return net

X = numpy.loadtxt("poco_1.prn", skiprows=11, usecols=(1, 2, 3, 4, 5, 6, 7), dtype=numpy.float32)
y_str = numpy.loadtxt("poco_1.prn", skiprows=11, usecols=8, dtype=numpy.str)
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(set(y_str)))
y = label_encoder.transform(y_str)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
nn1 = create_neural_network()
nn1.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
print("Simple dense network:")
nn1.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_train, y_train))
loss2, acc2 = nn1.evaluate(x_test, y_test, verbose=0)
print("Loss: ", loss2)
print("Accuracy: ", acc2)
print('\n')
print("Prunable network:")
nn = create_neural_network_prunable()
nn.build(x_train.shape)
for i in nn.layers:
	i.save_kernel()
	i.save_bias()
nn.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
print("Before pruning:")
nn.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_train, y_train))
loss1, acc1 = nn.evaluate(x_test, y_test, verbose=0)
print("Loss: ", loss1)
print("Accuracy: ", acc1)
print("After pruning:")
for i in nn.layers:
	i.restore_kernel()
	i.restore_bias()
nn.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_train, y_train))