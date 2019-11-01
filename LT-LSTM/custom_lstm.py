import tensorflow
from tensorflow.keras import layers

#Tries to enable dynamic memory allocation on GPUs
try:
	for i in tensorflow.config.experimental.list_physical_devices("GPU"):
		tensorflow.config.experimental.set_memory_growth(i, True)
except:
	print("Device dynamic memory allocation failed!")

class PrunnableLSTMCell(layers.LSTMCell):
    """Custom LSTM Class for prunning weights"""
    def __init__(self, *args, **kwargs):
        super(PrunnableLSTMCell, self).__init__(*args, **kwargs)
        
        self.trainable_channels = None
        self.trainable_recurrent_channels = None
        self.trainable_bias = None
        
        self._kernel1 = None
        self._kernel2 = None
        
        self._recurrent_kernel1 = None
        self._recurrent_kernel2 = None
        
        self._bias1 = None
        self._bias2 = None
        
        self.saved_W = None
        self.saved_recW = None
        self.saved_bias = None
        
    def build(self, input_shape):
        input_dim = input_shape[-1]

        if type(self.recurrent_initializer).__name__ == 'Identity':
            def recurrent_identity(shape, gain=1., dtype=None):
                del dtype
                return gain * np.concatenate(
                    [np.identity(shape[0])] * (shape[1] // shape[0]), axis=1)

            self.recurrent_initializer = recurrent_identity

        self._kernel1 = self.add_weight(shape=(input_dim, self.units * 4),name='kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        
        self._recurrent_kernel1 = self.add_weight(
                                        shape=(self.units, self.units * 4),
                                        name='recurrent_kernel',
                                        initializer=self.recurrent_initializer,
                                        regularizer=self.recurrent_regularizer,
                                        constraint=self.recurrent_constraint)
        
        self.trainable_channels = tensorflow.ones((input_dim, self.units*4), dtype=tensorflow.uint8)
        self.trainable_recurrent_channels = tensorflow.ones((input_dim, self.units*4), dtype=tensorflow.uint8)
        
        self._kernel2 = tensorflow.zeros((input_dim, self.units*4))
        self._recurrent_kernel2 = tensorflow.zeros((input_dim, self.units*4))
        
        
        if self.use_bias:
            if self.unit_forget_bias:
                @K.eager
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self._bias1 = self.add_weight(shape=(self.units * 4,),      #inicializa bias1
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self._bias2 = tensorflow.zeros((self.units * 4,))     #inicializa bias2
        else:
            self._bias1 = None

        self._kernel1_i = self._kernel1[:, :self.units]
        self._kernel1_f = self._kernel1[:, self.units: self.units * 2]
        self._kernel1_c = self._kernel1[:, self.units * 2: self.units * 3]
        self._kernel1_o = self._kernel1[:, self.units * 3:]

        self._recurrent_kernel1_i = self._recurrent_kernel1[:, :self.units]
        self._recurrent_kernel1_f = (self._recurrent_kernel1[:, self.units: self.units * 2])
        self._recurrent_kernel1_c = (self._recurrent_kernel1[:, self.units * 2: self.units * 3])
        self._recurrent_kernel1_o = self._recurrent_kernel1[:, self.units * 3:]

        if self.use_bias:
            self._bias1_i = self._bias1[:self.units]
            self._bias1_f = self._bias1[self.units: self.units * 2]
            self._bias1_c = self._bias1[self.units * 2: self.units * 3]
            self._bias1_o = self._bias1[self.units * 3:]
        else:
            self._bias1_i = None
            self._bias1_f = None
            self._bias1_c = None
            self._bias1_o = None
        self.built = True
    
    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self._kernel1_i)
            x_f = K.dot(inputs_f, self._kernel1_f)
            x_c = K.dot(inputs_c, self._kernel1_c)
            x_o = K.dot(inputs_o, self._kernel1_o)
            if self.use_bias:
                x_i = K.bias_add(x_i, self._bias1_i)
                x_f = K.bias_add(x_f, self._bias1_f)
                x_c = K.bias_add(x_c, self._bias1_c)
                x_o = K.bias_add(x_o, self._bias1_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
                                                      self._recurrent_kernel1_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
                                                      self._recurrent_kernel1_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                            self._recurrent_kernel1_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
                                                      self._recurrent_kernel1_o))
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self._recurrent_kernel1)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]
        