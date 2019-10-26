import numpy
import tensorflow
from sklearn import preprocessing, model_selection
from lottery_ticket_hypothesis import PrunableDense
from tensorflow import optimizers, initializers, losses
from tensorflow.keras import models, layers, activations

try:
	devices = tensorflow.config.experimental.list_physical_devices("GPU")
	tensorflow.config.experimental.set_memory_growth(devices[0], True)
except:
	print("Device config failed!")

def create_neural_network_prunable():
	"""Prunable model of a fully-conected multilayer perceptron"""
	net = models.Sequential()
	net.add(PrunableDense(256, activation=activations.softsign, name="Dense0", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(128, activation=activations.softsign, name="Dense1", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(64, activation=activations.softsign, name="Dense2", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(32, activation=activations.softsign, name="Dense3", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(1, activation=activations.tanh, name="Output", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	return net

#Data reading and train/test saparation
X = numpy.loadtxt("poco_1.prn", skiprows=11, usecols=(1, 2, 3, 4, 5, 6, 7), dtype=numpy.float32)
y_str = numpy.loadtxt("poco_1.prn", skiprows=11, usecols=8, dtype=numpy.str)
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(set(y_str)))
y = label_encoder.transform(y_str)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

#Building, saving initial values of kernel/bias, compiling and training model
print("Prunable network:")
nn = create_neural_network_prunable()
nn.build(x_train.shape)
for i in nn.layers:
	i.save_kernel()
	i.save_bias()
nn.summary()
nn.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
print("Before pruning:")
nn.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_train, y_train))
loss, accuracy = nn.evaluate(x_test, y_test, verbose=0)
print("Loss:", loss)
print("Accuracy:", accuracy)

#Creating list of weights, saving weight value and index
#TODO: optimize?
l = []
for i in range(len(nn.layers)):
	t1, t2 = nn.layers[i].kernel, nn.layers[i].bias
	for j in range(t1.shape[0]):
		for k in range(t1[j].shape[0]):
			l.append((t1[j][k].numpy(), i, j, k))
	for j in range(t2.shape[0]):
		l.append((t2[j], i, j))

#Sorting weights for pruning
#TODO: change p to p**(1/n)
s = sorted(l, key=lambda x: x[0])
p = int(numpy.floor((9. / 10.) * len(s)))
s = s[:p]
del l

#Creating dictionaries for separating weights and bias by layer
to_prune_dict = {}
to_prune_dict_kernel, to_prune_dict_bias = {}, {}
for i in range(len(nn.layers)):
	to_prune_dict_kernel[i] = []
	to_prune_dict_bias[i] = []
for i in s:
	if len(i) > 2:
		to_prune_dict_kernel[i[1]].append(i[2:])
	else:
		to_prune_dict_bias[i[1]].append(i[2])

#Creating pruning Tensor for pruning weights and pruning each layer
for i in to_prune_dict_kernel.keys():
	t = tensorflow.Variable(numpy.ones(nn.layers[i].kernel.shape))
	for j in to_prune_dict_kernel[i]:
		t[j[0], j[1]].assign(0)
	nn.layers[i].prune_kernel(t)

#Creating pruning Tesor for bias and pruning each layer
for i in to_prune_dict_bias.keys():
	t = tensorflow.Variable(numpy.ones(nn.layers[i].bias.shape))
	for j in to_prune_dict_bias[i]:
		t[j].assign(0)
	nn.layers[i].prune_bias(t)

#Restoring initial values
for i in nn.layers:
	i.restore_kernel()
	i.restore_bias()

#Training again
print("After pruning:")
nn.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_train, y_train))
loss, accuracy = nn.evaluate(x_test, y_test, verbose=0)
print("Loss:", loss)
print("Accuracy:", accuracy)

#Printing active edges
t = tensorflow.Variable(0, dtype=tensorflow.int64)
for i in nn.layers:
	j = tensorflow.math.count_nonzero(i.trainable_channels)
	t.assign_add(j)
	print(i.name, ":", j)
print("Active edges:", t.numpy())

#Printing disabled edges
t.assign(0)
for i in nn.layers:
	j = tensorflow.reduce_sum(tensorflow.cast((i.trainable_channels == 0), dtype=tensorflow.int64))
	t.assign_add(j)
	print(i.name, ":", j)
print("Disabled edges:", t.numpy())
