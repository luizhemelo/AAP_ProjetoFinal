import tensorflow
from tensorflow.keras import layers

class PrunnableLSTM(layers.LSTM):
    """Custom LSTM Class for prunning weights"""
    def __init__(self, *args, **kwargs):
		super(PrunnableLSTM, self).__init__(*args, **kwargs)
		self.trainable_channels = None
		self.trainable_bias = None
		self._kernel1 = None
		self._bias1 = None
		self.saved_W = None
		self.saved_bias = None