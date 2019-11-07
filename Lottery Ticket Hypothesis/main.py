import numpy
import tensorflow
from sklearn import preprocessing, model_selection
from lottery_ticket_hypothesis import PrunableDense
from tensorflow.keras import models, layers, activations
from tensorflow import optimizers, initializers, losses, metrics


#Tries to enable dynamic memory allocation on GPUs
try:
	for i in tensorflow.config.experimental.list_physical_devices("GPU"):
		tensorflow.config.experimental.set_memory_growth(i, True)
except:
	print("Device dynamic memory allocation failed!")

def create_neural_network_prunable():
	"""Prunable model of a fully-connected multilayer perceptron"""
	net = models.Sequential()
	net.add(PrunableDense(256, activation=activations.softsign, name="Dense0", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(128, activation=activations.softsign, name="Dense1", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(64, activation=activations.softsign, name="Dense2", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(32, activation=activations.softsign, name="Dense3", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	net.add(PrunableDense(1, activation=activations.tanh, name="Output", bias_initializer=tensorflow.ones, kernel_initializer=initializers.he_normal()))
	return net

#Data reading and train/test separation
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
nn.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=losses.BinaryCrossentropy(), metrics=[metrics.BinaryAccuracy()])
print("Before pruning:")
nn.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_train, y_train))
loss, accuracy = nn.evaluate(x_test, y_test, verbose=0)
print("Loss:", loss)
print("Accuracy:", accuracy)

#Creating tensor of weights
l1, l2 = [], []
for layer in nn.layers:
	l1.append(tensorflow.reshape(layer.kernel, (-1,)))
	l1.append(tensorflow.reshape(layer.bias, (-1,)))
	l2.append(tensorflow.reshape(layer.trainable_channels, (-1,)))
	l2.append(tensorflow.reshape(layer.trainable_bias, (-1,)))

#Sorting and getting threshold
s = tensorflow.sort(tensorflow.concat(l1, axis=-1))[tensorflow.concat(l2, axis=-1) == 1]
p = int(numpy.floor((9. / 10.) * len(s)))
threshold = s[p].numpy()

#Pruning each layer kernel and bias
for layer in nn.layers:
	indices_kernel = tensorflow.where(layer.kernel < threshold)
	indices_bias = tensorflow.where(layer.bias < threshold)
	t1 = tensorflow.tensor_scatter_nd_update(tensorflow.ones(layer.kernel.shape), indices_kernel, tensorflow.zeros(len(indices_kernel)))
	t2 = tensorflow.tensor_scatter_nd_update(tensorflow.ones(layer.bias.shape), indices_bias, tensorflow.zeros(len(indices_bias)))
	layer.prune_kernel(t1)
	layer.prune_bias(t2)

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
print("Active edges:")
t = tensorflow.Variable(0, dtype=tensorflow.int64)
for i in nn.layers:
	j = tensorflow.math.count_nonzero(i.trainable_channels)
	t.assign_add(j)
	print(i.name, ":", j)
print("Total:", t.numpy())
print()

#Printing disabled edges
t.assign(0)
print("Disabled edges:")
for i in nn.layers:
	j = tensorflow.reduce_sum(tensorflow.cast((i.trainable_channels == 0), dtype=tensorflow.int64))
	t.assign_add(j)
	print(i.name, ":", j)
print("Total:", t.numpy())
