"""
Helper classes for encoding sequences.

"""
import tensorflow as tf


class SequenceEncoder:
    """
    Abstract class for encoding sequences [B, T, input_units] into sequences [B, T, output_units].
    """

    def __init__(self):
        pass

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, self.name)

    def __call__(self, inputs, lengths=None):
        """

        :param inputs: [B, T, d]
        :param lengths: [B]
        :return: outputs [B, T, output_units]
        """
        pass


class EmbeddingEncoder(SequenceEncoder):
    """
    This encoder simply embeds integers in the sequence.

    inputs: [B, T]
    outputs: [B, T, num_units]

    """

    def __init__(self, num_units: int, vocab_size: int, reuse=None, name='EmbeddingEncoder'):
        self.output_units = num_units
        self.vocab_size = vocab_size
        self.name = name
        self.reuse = reuse

    def __call__(self, inputs, lengths=None):
        """

        :param inputs: [B, T]
        :param lengths: [B]
        :return: outputs [B, T, output_units]
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            # [V, d]
            E = tf.get_variable(
                name='E', initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self.vocab_size, self.output_units])
            # [B, M, output_units]
            outputs = tf.nn.embedding_lookup(
                E,  # [V, d]
                inputs  # [B, T]
            )
            return outputs


class UnidirectionalEncoder(SequenceEncoder):
    """
    This encoder is a forward RNN.
    """

    def __init__(self, num_units: int, cell_type='lstm', reuse=None, name='UnidirectionalEncoder'):
        self._num_units = num_units
        self._cell_type = cell_type
        self.output_units = num_units
        self.name = name
        self.reuse = reuse

    def __call__(self, inputs, lengths=None):
        """

        :param inputs: [B, T, input_dim]
        :param lengths: [B]
        :return: outputs [B, T, output_units], states [B, T, output_units]
        """
        if self._cell_type == 'lstm':
            cell_class = tf.contrib.rnn.BasicLSTMCell
        elif self._cell_type == 'gru':
            cell_class = tf.contrib.rnn.GRUCell
        else:
            raise ValueError('Unknown cell_type=%s' % self._cell_type)

        with tf.variable_scope(self.name, reuse=self.reuse):
            cell = cell_class(num_units=self._num_units)
            outputs, states = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs,
                sequence_length=lengths,
                dtype=tf.float32
            )
        return outputs  #, states


class PrefixEncoder(SequenceEncoder):
    """
    This encoder is a forward RNN.
    """

    def __init__(self, num_units: int, cell_type='lstm', reuse=None, name='PrefixEncoder'):
        #self._input_units = input_units
        self._num_units = num_units
        self._cell_type = cell_type
        self.output_units = num_units
        self.name = name
        self.reuse = reuse

    def __call__(self, inputs, lengths=None):
        """

        :param inputs: [B, T, input_dim]
        :param lengths: [B]
        :return: outputs [B, T, output_units], states [B, T, output_units]
        """
        if self._cell_type == 'lstm':
            cell_class = tf.contrib.rnn.BasicLSTMCell
        elif self._cell_type == 'gru':
            cell_class = tf.contrib.rnn.GRUCell
        else:
            raise ValueError('Unknown cell_type=%s' % self._cell_type)

        batch_size = tf.shape(inputs)[0]
        input_dim = tf.shape(inputs)[2]

        # adds an initial time step (represented by zeros)
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])

        with tf.variable_scope(self.name, reuse=self.reuse):
            cell = cell_class(num_units=self._num_units)
            outputs, states = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs,
                sequence_length=lengths,
                dtype=tf.float32
            )

        # remove the last time step
        outputs = outputs[:, :-1, :]

        return outputs  #, states


class BidirectionalEncoder(SequenceEncoder):
    """
    This encoder is a bidirectional RNN.

    """

    def __init__(self, num_units: int,
                 cell_type='lstm',
                 merge_strategy='sum',
                 dropout=False,
                 input_keep_prob=1.0,
                 output_keep_prob=1.0,
                 state_keep_prob=1.0,
                 variational_recurrent=False,
                 reuse=None,
                 name='BidirectionalEncoder'):
        self._num_units = num_units
        self._cell_type = cell_type
        self._merge_strategy = merge_strategy
        self._dropout = dropout
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._state_keep_prob = state_keep_prob
        self._variational_recurrent = variational_recurrent
        if merge_strategy not in ['sum', 'concat']:
            raise ValueError('Unknown merge_strategy=%s' % merge_strategy)
        self.output_units = num_units if merge_strategy == 'sum' else 2 * num_units
        self.name = name
        self.reuse = reuse

    def __call__(self, inputs, lengths=None):
        """

        :param inputs: [B, T, input_dim]
        :param lengths: [B]
        :return: outputs [B, T, output_units], states [B, T, output_units]
        """
        if self._cell_type == 'lstm':
            cell_class = tf.contrib.rnn.BasicLSTMCell
        elif self._cell_type == 'gru':
            cell_class = tf.contrib.rnn.GRUCell
        else:
            raise ValueError('Unknown cell_type=%s' % self._cell_type)

        with tf.variable_scope(self.name, reuse=self.reuse):
            cell_fw = cell_class(num_units=self._num_units)
            cell_bw = cell_class(num_units=self._num_units)

            if self._dropout:
                cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell_fw,
                    input_keep_prob=self._input_keep_prob,
                    output_keep_prob=self._output_keep_prob,
                    state_keep_prob=self._state_keep_prob,
                    variational_recurrent=self._variational_recurrent,
                    dtype=inputs.dtype,
                    input_size=lengths  # [B]
                )
                cell_bw = tf.contrib.rnn.DropoutWrapper(
                    cell_bw,
                    input_keep_prob=self._input_keep_prob,
                    output_keep_prob=self._output_keep_prob,
                    state_keep_prob=self._state_keep_prob,
                    variational_recurrent=self._variational_recurrent,
                    dtype=inputs.dtype,
                    input_size=lengths  # [B]
                )

            (outputs_fw, outputs_bw), (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                dtype=tf.float32
            )

            if self._merge_strategy == 'sum':
                # [B, T, num_units]
                outputs = tf.add(outputs_fw, outputs_bw)
                states = tf.add(states_fw, states_bw)
            else:  # concat
                # [B, T, num_units *  2]  (but note we redifined dh)
                outputs = tf.concat([outputs_fw, outputs_bw], -1)
                states = tf.concat([states_fw, states_bw], -1)

        return outputs  #, states


class MultiBidirectionalEncoder(SequenceEncoder):
    """
    This encoder is a bidirectional RNN.

    """

    def __init__(self, num_units: int, num_layers: int, cell_type='lstm', merge_strategy='sum', reuse=None, name='BidirectionalEncoder'):
        self._num_layers = num_layers
        self._num_units = num_units
        self._cell_type = cell_type
        self._merge_strategy = merge_strategy
        if merge_strategy not in ['sum', 'concat']:
            raise ValueError('Unknown merge_strategy=%s' % merge_strategy)
        self.output_units = num_units if merge_strategy == 'sum' else 2 * num_units
        self.name = name
        self.reuse = reuse

    def __call__(self, inputs, lengths=None):
        """

        :param inputs: [B, T, input_dim]
        :param lengths: [B]
        :return: outputs [B, T, output_units], states [B, T, output_units]
        """
        if self._cell_type == 'lstm':
            cell_class = tf.contrib.rnn.BasicLSTMCell
        elif self._cell_type == 'gru':
            cell_class = tf.contrib.rnn.GRUCell
        else:
            raise ValueError('Unknown cell_type=%s' % self._cell_type)

        with tf.variable_scope(self.name, reuse=self.reuse):
            # first we create all fwd/bwd celss
            fw_cells = []
            bw_cells = []
            for i in range(self._num_layers):
                fw_cells.append(cell_class(num_units=self._num_units))
                bw_cells.append(cell_class(num_units=self._num_units))
            # then we wrap them around MultiRNNCell
            cell_fw = tf.contrib.rnn.MultiRNNCell(fw_cells)
            cell_bw = tf.contrib.rnn.MultiRNNCell(bw_cells)
            # and dynamically process the inputs
            (outputs_fw, outputs_bw), (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                dtype=tf.float32
            )

            if self._merge_strategy == 'sum':
                # [B, T, num_units]
                outputs = tf.add(outputs_fw, outputs_bw)
                states = tf.add(states_fw, states_bw)
            else:  # concat
                # [B, T, num_units *  2]  (but note we redifined dh)
                outputs = tf.concat([outputs_fw, outputs_bw], -1)
                states = tf.concat([states_fw, states_bw], -1)

        return outputs  #, states


class PassThrough(SequenceEncoder):
    """
    This is a dummy encoder that simply passes the inputs through.

    """

    def __init__(self, input_units: int, reuse=None, name='PassThrough'):
        """

        :param input_units: number of input (also output) units
        :param name:
        """
        self.output_units = input_units
        self.name = name
        self.reuse = reuse

    def __call__(self, inputs, lengths=None):
        """

        :param inputs: [B, T, input_units]
        :param lengths: [B]
        :return: outputs=inputs [B, T, output_units=input_units]
        """
        return inputs


class FeedForwardEncoder(SequenceEncoder):
    """
    This encoder is simply a time-distributed feed forward neural network.

    """

    def __init__(self, num_units: int, activation_fn=None, hidden_layers=[], reuse=None, name='FeedForwardEncoder'):
        """

        :param num_units: number of output units in the top layer
        :param activation_fn: activation function of the top layer
        :param hidden_layers: a bottom-up sequence of hidden layers each expressed as
            - a tuple (num_units, activation_fn)
            - or a FeedForwardEncoder object
        """
        self._num_units = num_units
        self._activation_fn = activation_fn
        self._hidden_layers = hidden_layers
        self.output_units = num_units
        self.name = name
        self.reuse = reuse

    @classmethod
    def construct(cls, layers: list, name, reuse=None) -> 'FeedForwardEncoder':
        """
        Construct a FeedForwardEncoder from a list of pairs specifying layers.
        :param layers: list of pairs of kind (num_units, activation_fn)
        :return: FeedForwardEncoder
        """
        if not layers:
            raise ValueError('Need a non-empty list of pairs (num_units, activation_fn)')
        return FeedForwardEncoder(
            num_units=layers[-1][0],
            activation_fn=layers[-1][1],
            hidden_layers=layers[:-1],
            reuse=reuse,
            name=name)

    def __call__(self, inputs, lengths=None):
        """

        :param inputs: [B, T, input_dim]
        :param lengths: [B]
        :return: [B, T, output_units]
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            h = inputs
            for i, layer in enumerate(self._hidden_layers):
                if type(layer) in [tuple, list]:
                    h = tf.contrib.layers.fully_connected(
                        h,
                        num_outputs=layer[0],
                        activation_fn=layer[1]
                    )
                elif type(layer) is FeedForwardEncoder:
                    h = layer.predict(h)
                else:
                    raise ValueError('Unknown type of layer: %s' % type(layer))
            outputs = tf.contrib.layers.fully_connected(
                h,
                num_outputs=self._num_units,
                activation_fn=self._activation_fn
            )
            return outputs


class StackedEncoder(SequenceEncoder):

    def __init__(self, encoders: list, reuse=None, name='StackedEncoder'):
        self._encoders = encoders
        self.output_units = encoders[-1].output_units
        self.name = name
        self.reuse = reuse

    def __call__(self, inputs, lengths=None):
        h = inputs
        for encoder in self._encoders:
            h = encoder(inputs=h, lengths=lengths)
        return h

    @classmethod
    def construct(cls, layers: list, name, reuse=None) -> 'StackedEncoder':
        """
        Construct from a list of layers specified in a simpler way.
            Each layer is a tuple, the first element is one of
                - ffnn => FeedForwardEncoder
                - rnn => UnidirectionalEncoder
                - birnn => BidirectionalEncoder
                - prefix => PrefixEncoder
            the second element is always the number of units
            the third element is
                - activation_fn for ffnn
                - cell_type for rnn/birrn/prefix
            and a fourth element required for birnn is
                - merge_strategy
        :param layers:
        :param name:
        :return:
        """
        encoders = []
        for i, layer in enumerate(layers):
            name_i = '%s-%d-%s' % (name, i, layer[0])
            if layer[0] == 'ffnn':
                encoders.append(FeedForwardEncoder(
                    num_units=layer[1],
                    activation_fn=layer[2],
                    reuse=reuse,
                    name=name_i
                ))
            elif layer[0] == 'rnn':
                encoders.append(UnidirectionalEncoder(
                    num_units=layer[1],
                    cell_type=layer[2],
                    reuse=reuse,
                    name=name_i
                ))
            elif layer[0] == 'birnn':
                encoders.append(BidirectionalEncoder(
                    num_units=layer[1],
                    cell_type=layer[2],
                    merge_strategy=layer[3],
                    reuse=reuse,
                    name=name_i
                ))
            elif layer[0] == 'prefix':
                encoders.append(PrefixEncoder(
                    num_units=layer[1],
                    cell_type=layer[2],
                    reuse=reuse,
                    name=name_i
                ))
        return StackedEncoder(encoders, reuse, name)


def ffnn(
        inputs,
        layers
):
    """
    A FFNN.

    :param inputs: [B, dim]
        - collapse sample and time dimension if your inputs are sequences
    :param layers: list of pairs (num_outputs, activation_fn)
    :return: [B, num_outputs]
    """
    if not layers:
        return inputs

    h = tf.contrib.layers.fully_connected(
        inputs,  # [B * T, dim]
        num_outputs=layers[0][0],
        activation_fn=layers[0][1]
    )

    for num_outputs, activation_fn in layers[1:]:
        h = tf.contrib.layers.fully_connected(
            h,
            num_outputs=num_outputs,
            activation_fn=activation_fn
        )

    # [B, num_outputs]
    return h


def encode_embedded_sequences(
        inputs,  # [B, T, dx]
        input_dim,  # dx
        num_units,  # dh
        rnn_cell=None,
        merge_strategy='sum',
        mlp_nonlinearity=tf.nn.tanh,
        dtype=tf.float32
):
    """
    A (possibly bi-directional) encoder for embedded sequences.
    
    :param inputs: embedded sequences [B, T, dx] 
    :param input_dim: embedding dimensionality dx
    :param num_units: hidden units 
    :param rnn_cell: type of RNN cell (e.g. 'gru', 'lstm', or None)
    :param merge_strategy: how to merge fwd and bwd rnn states (e.g. 'sum', 'concat', 'mlp')
        - note that if you choose 'concat' there will be 2*num_units hidden units
    :param mlp_nonlinearity: nonlinearity for all MLPs involved
    :param dtype: data type for RNNs
    :return: encoddings [B, T, dh], dh 
        - dh = num_units unless rnn_cell is not None and merge is 'concat' in which case dh = num_units * 2
    """
    batch_size = tf.shape(inputs)[0]
    longest = tf.shape(inputs)[1]
    if rnn_cell is not None:  # here we use bidirectional encodings of x_1^m
        if rnn_cell == 'gru':
            rnn_fw = tf.contrib.rnn.GRUCell(num_units=num_units)
            rnn_bw = tf.contrib.rnn.GRUCell(num_units=num_units)
        elif rnn_cell == 'lstm':
            rnn_fw = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
            rnn_bw = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
        else:
            raise ValueError('Unknown RNN cell type: %s' % rnn_cell)
        bi_rnn, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_fw,
            cell_bw=rnn_bw,
            inputs=inputs,
            dtype=dtype
        )
        # Merge sum
        if merge_strategy == 'sum':
            # [B, T, num_units]
            h = tf.add(bi_rnn[0], bi_rnn[1])
            dh = num_units
        elif merge_strategy == 'concat':
            # [B, T, num_units *  2]  (but note we redifined dh)
            h = tf.concat([bi_rnn[0], bi_rnn[1]], -1)
            dh = num_units * 2
        elif merge_strategy == 'mlp':
            # [B, T, num_units * 2]  (but note we redifined dh)
            h = tf.concat([bi_rnn[0], bi_rnn[1]], -1)
            # [B * T, num_units]
            h = tf.contrib.layers.fully_connected(
                tf.reshape(h, shape=[batch_size * longest, num_units * 2]),
                num_outputs=num_units,
                activation_fn=mlp_nonlinearity
            )
            # [B * T, num_units]
            h = tf.contrib.layers.fully_connected(
                tf.reshape(h, shape=[batch_size * longest, num_units]),
                num_outputs=num_units,
                activation_fn=None
            )
            dh = num_units
        else:
            raise ValueError('Unknown BiRNN merge strategy: %s' % merge_strategy)
    else:  # here we use the embeddings of x_1^m
        # one-layer hidden encoder
        # [B * M, dh]
        h = tf.contrib.layers.fully_connected(
            tf.reshape(inputs, shape=[batch_size * longest, input_dim]),
            num_outputs=num_units,
            activation_fn=mlp_nonlinearity
        )
        dh = num_units
    return tf.reshape(h, shape=[batch_size, longest, dh]), dh


def encode_embedded_prefixes(
        inputs,  # [B, T, dx]
        num_units,  # dh
        lengths=None,  # [B]
        cell_class=tf.contrib.rnn.BasicLSTMCell,
        dtype=tf.float32
):
    """
    A (possibly bi-directional) encoder for embedded sequences.

    :param inputs: embedded sequences [B, T, dx]
    :param input_dim: embedding dimensionality dx
    :param num_units: hidden units
    :param cell_class: type of RNN cell (e.g. 'gru', 'lstm', or None)
    :param merge_strategy: how to merge fwd and bwd rnn states (e.g. 'sum', 'concat', 'mlp')
        - note that if you choose 'concat' there will be 2*num_units hidden units
    :param mlp_nonlinearity: nonlinearity for all MLPs involved
    :param dtype: data type for RNNs
    :return: encoddings [B, T, dh], dh
        - dh = num_units unless rnn_cell is not None and merge is 'concat' in which case dh = num_units * 2
    """
    cell = cell_class(num_units=num_units)
    outputs, states = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=inputs,
        sequence_length=lengths,
        dtype=dtype
    )
    return outputs, states


def cnn_encode_embedded_sequences(
        inputs,  # [B, T, dx]
        num_units, #dh
        # cnn
        filters=2,
        kernel_size=2,
        cnn_nonlinearity=tf.nn.tanh,
        padding='same', # [B, T, dx]
        #extra cnn
        deep_cnn=[], #define each cnn+pooling layer
        # pool
        pool_size=2,
        strides=2,
):
    """
    
    :param inputs: inputs: embedded sequences [B, T, dx] 
    :param num_units: 
    :param filters: 
    :param kernel_size: 
    :param cnn_nonlinearity: 
    :param padding: 
    :param pool_size: 
    :param strides: 
    :return: 
    """

    batch_size = tf.shape(inputs)[0]
    longest = tf.shape(inputs)[1]
    # Convolutional Layer #1 over time 1d
    # [B, T, dx]
    conv1 = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=cnn_nonlinearity)

    # Pooling Layer #1 Max
    # [B, T, dx]
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=pool_size, strides=strides)

    # TODO add 2d cnn layers
    #conv2 = tf.layers.conv1d(
    #    inputs=pool1,
    #    filters=filters,
    #    kernel_size=kernel_size,
    #    padding=padding,
    #    activation=cnn_nonlinearity)

    # Pooling Layer #1 Max
    #pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=pool_size, strides=strides)

    # [B, T, dh]
    # fully conected layer with dh units
    h = tf.contrib.layers.fully_connected(
        tf.reshape(pool1, shape=[batch_size * longest, num_units]), # TODO check correct size of pool output
        num_outputs=num_units,
        activation_fn=None
    )
    dh = num_units
    return tf.reshape(h, shape=[batch_size, longest, dh]), dh
