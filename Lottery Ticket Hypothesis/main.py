import numpy
import tensorflow
from tensorflow.keras import models, layers
from sklearn import preprocessing, model_selection
from lottery_ticket_hypothesis import PrunableDense
from tensorflow import optimizers, config, initializers, losses

devices = tensorflow.config.experimental.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(devices[0], True)

def create_neural_network():
	net = models.Sequential()
	net.add(layers.Dense(256, activation="softsign", name="Dense0", bias_initializer="ones", kernel_initializer="he_normal"))
	net.add(layers.Dense(128, activation="softsign", name="Dense1", bias_initializer="ones", kernel_initializer="he_normal"))
	net.add(layers.Dense(64, activation="softsign", name="Dense2", bias_initializer="ones", kernel_initializer="he_normal"))
	net.add(layers.Dense(32, activation="softsign", name="Dense3", bias_initializer="ones", kernel_initializer="he_normal"))
	net.add(layers.Dense(1, activation="tanh", name="Output", kernel_initializer="he_normal", bias_initializer="ones"))
	return net

def create_neural_network_pruned():
	net = models.Sequential()
	net.add(PrunableDense(256, activation="softsign", name="Dense0", bias_initializer="ones", kernel_initializer="he_normal"))
	net.add(PrunableDense(128, activation="softsign", name="Dense1", bias_initializer="ones", kernel_initializer="he_normal"))
	net.add(PrunableDense(64, activation="softsign", name="Dense2", bias_initializer="ones", kernel_initializer="he_normal"))
	net.add(PrunableDense(32, activation="softsign", name="Dense3", bias_initializer="ones", kernel_initializer="he_normal"))
	net.add(PrunableDense(1, activation="tanh", name="Output", bias_initializer="ones"))
	return net

X = numpy.loadtxt("poco_1.prn", skiprows=11, usecols=(1, 2, 3, 4, 5, 6, 7), dtype=numpy.float32)
y = numpy.loadtxt("poco_1.prn", skiprows=11, usecols=8, dtype=numpy.str)
le = preprocessing.LabelEncoder()
le.fit(list(set(y)))
y_t = le.transform(y)
classes = ["0", "1"]
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y_t, test_size=0.33, random_state=42)
nn = create_neural_network_pruned()
nn.compile(optimizer=optimizers.Adam(learning_rate=0.00001), loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
nn.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_train, y_train))
loss1, acc1 = nn.evaluate(x_test, y_test, verbose=0)
print("Loss: ", loss1)
print("Accuracy: ", acc1)
nn1 = create_neural_network()
nn1.compile(optimizer=optimizers.Adam(learning_rate=0.00001), loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
nn1.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_train, y_train))
loss2, acc2 = nn1.evaluate(x_test, y_test, verbose=0)
print("Loss: ", loss2)
print("Accuracy: ", acc2)