import theano
import theano.tensor as T

import lasagne
# from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers import Conv2DLayer, Conv3DLayer
from lasagne.layers import MaxPool2DLayer, MaxPool3DLayer, InputLayer, MaxPool1DLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer
from lasagne.layers import batch_norm


def build_cnn(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=[32, 32], n_colors=3,
              isMaxpool=True, filter_size=(3, 3), dropout=0.0, batch_norm_conv=False):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)

    :param input_var: theano variable as input to the network
    :param w_init: Initial weight values
    :param n_layers: number of layers in each stack. An array of integers with each
                    value corresponding to the number of layers hesain each stack.
                    (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
    :param n_filters_first: number of filters in the first layer
    :param imSize: Size of the image
    :param n_colors: Number of color channels (depth)
    :return: a pointer to the output of last layer
    """
    weights = []  # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if w_init is None:
        w_init = [lasagne.init.GlorotUniform(gain="relu")] * sum(n_layers)
    # Input layer
    network = InputLayer(shape=(None, n_colors, imsize[0], imsize[1]),
                         input_var=input_var)

    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv2DLayer(network, num_filters=n_filters_first * (2 ** i), filter_size=filter_size,
                                  W=w_init[count], pad='same')
            count += 1
            weights.append(network.W)
            if batch_norm_conv:
                network = batch_norm(network)
        if isMaxpool:
            network = MaxPool2DLayer(network, pool_size=(2, 2))
            if dropout:
                network = lasagne.layers.dropout(network, p=dropout)
    return network, weights


def build_cnn_basic(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=[32, 32], n_colors=3,
              isMaxpool=True, filter_size=((2, 2), (2, 2), (1, 2)), dropout=0.0, batch_norm_conv=False):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)

    :param input_var: theano variable as input to the network
    :param w_init: Initial weight values
    :param n_layers: number of layers in each stack. An array of integers with each
                    value corresponding to the number of layers hesain each stack.
                    (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
    :param n_filters_first: number of filters in the first layer
    :param imSize: Size of the image
    :param n_colors: Number of color channels (depth)
    :return: a pointer to the output of last layer
    """
    weights = []  # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if w_init is None:
        w_init = [lasagne.init.GlorotUniform()] * sum(n_layers)
    # Input layer
    network = InputLayer(shape=(None, n_colors, imsize[0], imsize[1]),
                         input_var=input_var)

    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv2DLayer(network, num_filters=n_filters_first * (2 ** i), filter_size=filter_size[l],
                                  W=w_init[count], pad='valid')
            count += 1
            weights.append(network.W)
            if batch_norm_conv:
                network = batch_norm(network)
        if isMaxpool:
            network = MaxPool2DLayer(network, pool_size=(2, 2))
            if dropout:
                network = lasagne.layers.dropout(network, p=dropout)
    return network, weights


def build_cnn3d(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=[32, 32], n_colors=3,
                n_timewin=5, isMaxpool=True, filter_size=[(3, 1, 1),(3, 3, 3),(3, 3, 3)], dropout=0.0, input_dropout=0.0,
                batch_norm_conv=False, padding='same', pool_size=[(2, 1, 1), (2, 2, 2), (2, 2, 2)], factor=2):
    """
    Builds a VGG style 3D CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)

    :param input_var: theano variable as input to the network
    :param w_init: Initial weight values
    :param n_layers: number of layers in each stack. An array of integers with each
                    value corresponding to the number of layers hesain each stack.
                    (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
    :param n_filters_first: number of filters in the first layer
    :param imSize: Size of the image
    :param n_colors: Number of color channels (depth)
    :return: a pointer to the output of last layer
    """
    weights = []  # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if w_init is None:
        w_init = [lasagne.init.GlorotUniform()] * sum(n_layers)
    # Input layer
    network = InputLayer(shape=(n_timewin, None, n_colors, imsize[0], imsize[1]), input_var=input_var)

    network = DimshuffleLayer(network, (1, 2, 0, 3, 4))

    if input_dropout:
        network = lasagne.layers.dropout(network, p=input_dropout)

    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv3DLayer(network, num_filters=n_filters_first * (factor ** i), filter_size=filter_size[i],
                                  W=w_init[count], pad=padding)
            count += 1
            weights.append(network.W)
            if dropout:
                network = lasagne.layers.dropout(network, p=dropout)
        if batch_norm_conv:
            network = batch_norm(network)
        if isMaxpool:
            network = MaxPool3DLayer(network, pool_size=pool_size[i])
    return network, weights


def build_cnn_int(input_network, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, filter_size=(3, 3),
                  pool_size=(2, 2), isMaxpool=True):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)

    :param input_var: theano variable as input to the network
    :param w_init: Initial weight values
    :param n_layers: number of layers in each stack. An array of integers with each
                    value corresponding to the number of layers in each stack.
                    (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
    :param n_filters_first: number of filters in the first layer
    :param imSize: Size of the image
    :param n_colors: Number of color channels (depth)
    :return: a pointer to the output of last layer
    """
    weights = []  # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if w_init is None:
        w_init = [lasagne.init.GlorotUniform()] * sum(n_layers)

    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv2DLayer(input_network, num_filters=n_filters_first * (2 ** i), filter_size=filter_size,
                                  W=w_init[count], pad='same')
            count += 1
            weights.append(network.W)
        if isMaxpool:
            network = MaxPool2DLayer(network, pool_size=pool_size)
    return network, weights

def network_convpool_cnn(input_vars, nb_classes, imsize=[32, 32], n_colors=3, n_layers=(4, 2), n_filters_first=32,
                         dense_num_unit=[512, 512], batch_norm_dense=False, batch_norm_conv=False):
    """
    Builds the complete network with maxpooling layer in time.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :return: a pointer to the output of last layer
    """

    convnet, _ = build_cnn(input_vars, imsize=imsize, n_colors=n_colors, n_filters_first=n_filters_first,
                           n_layers=n_layers, batch_norm_conv=batch_norm_conv)

    # convpool = FlattenLayer(convnet)
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    for i in range(len(dense_num_unit)):
        convnet = DenseLayer(lasagne.layers.dropout(convnet, p=.5), num_units=dense_num_unit[i],
                             nonlinearity=lasagne.nonlinearities.rectify)
        if batch_norm_dense:
            convnet = batch_norm(convnet)

    if nb_classes == 1:
        nonlinearity = lasagne.nonlinearities.sigmoid
    else:
        nonlinearity = lasagne.nonlinearities.softmax

    convnet = DenseLayer(lasagne.layers.dropout(convnet, p=.5), num_units=nb_classes, nonlinearity=nonlinearity)
    return convnet

def network_convpool_cnn_basic(input_vars, nb_classes, imsize=[32, 32], n_colors=3, n_layers=(4, 2), n_filters_first=32,
                               dense_num_unit=[256, 256], filter_size=(3, 3)):
    """
    Builds the complete network with maxpooling layer in time.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :return: a pointer to the output of last layer
    """

    convnet, _ = build_cnn_basic(input_vars, imsize=imsize, n_colors=n_colors, n_filters_first=n_filters_first,
                           n_layers=n_layers, input_dropout=0.0, filter_size=filter_size, isMaxpool=False)

    # convpool = FlattenLayer(convnet)
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    for i in range(len(dense_num_unit)):
        convnet = DenseLayer(lasagne.layers.dropout(convnet, p=.5), num_units=dense_num_unit[i],
                             nonlinearity=lasagne.nonlinearities.rectify)

    if nb_classes == 1:
        nonlinearity = lasagne.nonlinearities.sigmoid
    else:
        nonlinearity = lasagne.nonlinearities.softmax

    convnet = DenseLayer(lasagne.layers.dropout(convnet, p=.5), num_units=nb_classes, nonlinearity=nonlinearity)
    return convnet


def network_convpool_cnn3d(input_vars, nb_classes, imsize=[32, 32], n_colors=3, n_timewin=5, n_layers=(4, 2),
                           n_filters_first=32, dense_num_unit=[512,512], batch_norm_dense=False,
                           pool_size=[(2, 1, 1),(2, 2, 2), (2, 2, 2)], dropout_dense=True, batch_norm_conv=False,
                           filter_factor=2, filter_size=[(3, 1, 1),(3, 3, 3),(3, 3, 3)]):
    """
    Builds the complete network with maxpooling layer in time.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :return: a pointer to the output of last layer
    """

    convnet, _ = build_cnn3d(input_vars, imsize=imsize, n_colors=n_colors, n_timewin=n_timewin,
                             n_filters_first=n_filters_first, n_layers=n_layers, padding='same', isMaxpool=True,
                             pool_size=pool_size,batch_norm_conv=batch_norm_conv, factor=filter_factor,
                             filter_size=filter_size)

    convnet = FlattenLayer(convnet)
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    for i in range(len(dense_num_unit)):
        if dropout_dense:
            convnet=lasagne.layers.dropout(convnet, p=.5)
        convnet = DenseLayer(convnet, num_units=dense_num_unit[i],
                             nonlinearity=lasagne.nonlinearities.rectify)
        if batch_norm_dense:
            convnet = batch_norm(convnet)
    # And, finally, the 2-unit output layer with 50% dropout on its inputs:
    if nb_classes == 1:
        nonlinearity = lasagne.nonlinearities.sigmoid
    else:
        nonlinearity = lasagne.nonlinearities.softmax

    if dropout_dense:
        convnet = lasagne.layers.dropout(convnet, p=.5)
    convnet = DenseLayer(convnet, num_units=nb_classes, nonlinearity=nonlinearity)
    return convnet


def network_convpool_cnn_max(input_vars, nb_classes, imsize=[32, 32], n_colors=3, n_timewin=5, n_layers=(4, 2),
                             n_filters_first=32, dense_num_unit=[512, 512], shared_weights=True,
                             batch_norm_dense=False, batch_norm_conv=False):
    """
    Builds the complete network with maxpooling layer in time.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    # Build 5 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors,
                                        n_filters_first=n_filters_first, n_layers=n_layers,
                                        batch_norm_conv=batch_norm_conv)
            if not shared_weights:
                w_init = None
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors,
                                   n_filters_first=n_filters_first, n_layers=n_layers,
                                   batch_norm_conv=batch_norm_conv)
        convnets.append(FlattenLayer(convnet))
    # convpooling using Max pooling over frames
    convpool = ElemwiseMergeLayer(convnets, theano.tensor.maximum)
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    for i in range(len(dense_num_unit)):
        convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=dense_num_unit[i],
                              nonlinearity=lasagne.nonlinearities.rectify)
        if batch_norm_dense:
            convpool = batch_norm(convpool)
    # And, finally, the 2-unit output layer with 50% dropout on its inputs:
    if nb_classes == 1:
        nonlinearity = lasagne.nonlinearities.sigmoid
    else:
        nonlinearity = lasagne.nonlinearities.softmax

    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=nb_classes, nonlinearity=nonlinearity)

    return convpool


def network_convpool_conv1d(input_vars, nb_classes, imsize=[32, 32], n_colors=3, n_timewin=5, n_layers=(4, 2),
                            n_filters_first=32, dense_num_unit=[512, 512], shared_weights=True,
                            batch_norm_dense=False, batch_norm_conv=False):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    # Build 5 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors,
                                        n_filters_first=n_filters_first, n_layers=n_layers,
                                        batch_norm_conv = batch_norm_conv)
            if not shared_weights:
                w_init = None
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors,
                                   n_filters_first=n_filters_first, n_layers=n_layers,
                                   batch_norm_conv = batch_norm_conv)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    convpool = DimshuffleLayer(convpool, (0, 2, 1))
    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    convpool = Conv1DLayer(convpool, 64, 3)
    convpool = MaxPool1DLayer(convpool, pool_size=(2))
    convpool = Conv1DLayer(convpool, 128, 3)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    for i in range(len(dense_num_unit)):
        convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=dense_num_unit[i],
                              nonlinearity=lasagne.nonlinearities.rectify)
        if batch_norm_dense:
            convpool = batch_norm(convpool)
    # And, finally, the output layer with 50% dropout on its inputs:
    if nb_classes == 1:
        nonlinearity = lasagne.nonlinearities.sigmoid
    else:
        nonlinearity = lasagne.nonlinearities.softmax

    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                          num_units=nb_classes, nonlinearity=nonlinearity)
    return convpool


def network_convpool_lstm(input_vars, nb_classes, grad_clip=110, imsize=[32, 32], n_colors=3, n_timewin=5,
                          n_layers=(4, 2), n_filters_first=32, dense_num_unit=[512, 512],
                          shared_weights=True, batch_norm_dense=False, batch_norm_conv=False):
    """
    Builds the complete network with LSTM layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    # Build 5 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors,
                                        n_filters_first=n_filters_first, n_layers=n_layers,
                                        batch_norm_conv = batch_norm_conv)
            if not shared_weights:
                w_init = None
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors,
                                   n_filters_first=n_filters_first, n_layers=n_layers,
                                   batch_norm_conv = batch_norm_conv)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    convpool = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
                         nonlinearity=lasagne.nonlinearities.tanh)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.
    convpool = SliceLayer(convpool, -1, 1)  # Selecting the last prediction
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    for i in range(len(dense_num_unit)):
        convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=dense_num_unit[i],
                              nonlinearity=lasagne.nonlinearities.rectify)
        if batch_norm_dense:
            convpool = batch_norm(convpool)

    # And, finally, the output layer with 50% dropout on its inputs:
    if nb_classes == 1:
        nonlinearity = lasagne.nonlinearities.sigmoid
    else:
        nonlinearity = lasagne.nonlinearities.softmax
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                          num_units=nb_classes, nonlinearity=nonlinearity)
    return convpool


def network_convpool_mix(input_vars, nb_classes, grad_clip=110, imsize=[32, 32], n_colors=3, n_timewin=5,
                         n_layers=(4, 2), n_filters_first=32, dense_num_unit=[512, 512],
                          shared_weights=True, batch_norm_dense=False, batch_norm_conv=False):
    """
    Builds the complete network with LSTM and 1D-conv layers combined

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    # Build 5 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors,
                                        n_filters_first=n_filters_first, n_layers=n_layers,
                                        batch_norm_conv = batch_norm_conv)
            if not shared_weights:
                w_init = None
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors,
                                   n_filters_first=n_filters_first, n_layers=n_layers,
                                   batch_norm_conv = batch_norm_conv)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    reformConvpool = DimshuffleLayer(convpool, (0, 2, 1))
    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    conv_out = Conv1DLayer(reformConvpool, 64, 3)
    conv_out = FlattenLayer(conv_out)
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    lstm = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
                     nonlinearity=lasagne.nonlinearities.tanh)
    lstm_out = SliceLayer(lstm, -1, 1)
    # Merge 1D-Conv and LSTM outputs
    dense_input = ConcatLayer([conv_out, lstm_out])
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    for i in range(len(dense_num_unit)):
        convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5), num_units=dense_num_unit[i],
                              nonlinearity=lasagne.nonlinearities.rectify)
        if batch_norm_dense:
            convpool = batch_norm(convpool)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    if nb_classes == 1:
        nonlinearity = lasagne.nonlinearities.sigmoid
    else:
        nonlinearity = lasagne.nonlinearities.softmax
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                          num_units=nb_classes, nonlinearity=nonlinearity)
    return convpool


def network_convpool_lstm_hybrid(input_vars, nb_classes, grad_clip=110, imsize=[32, 32], n_colors=3, n_timewin=5,
                                 n_layers=(4, 2), n_filters_first=32, dense_num_unit=[512, 512],
                                 shared_weights=True, batch_norm_dense=False, batch_norm_conv=False):
    """
    Builds the complete network with LSTM and 1D-conv layers combined

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets_lstm = []
    convnets_first = []
    n_layers_first = [1]
    w_init = None
    # Build 5 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors,
                                        n_filters_first=n_filters_first, n_layers=n_layers, isMaxpool=False,
                                        filter_size=(2, 2), batch_norm_conv = batch_norm_conv)
            if not shared_weights:
                w_init = None
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors,
                                   n_filters_first=n_filters_first, n_layers=n_layers, isMaxpool=False,
                                   filter_size=(2, 2), batch_norm_conv = batch_norm_conv)
        convnets_lstm.append(FlattenLayer(convnet))
        convnets_first.append(convnet)

    # build lstm network
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets_lstm)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets_lstm[0])[1]))
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    lstm = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
                     nonlinearity=lasagne.nonlinearities.tanh)
    lstm_out = SliceLayer(lstm, -1, 1)

    # Build Second 5 parallel CNNs with shared weights
    w_init = None
    n_filters_second = 32
    n_layers_second = (3, 2)
    convnets_second = []
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn_int(convnets_first[i], n_filters_first=n_filters_second,
                                            n_layers=n_layers_second)
        else:
            convnet, _ = build_cnn_int(convnets_first[i], w_init=w_init, n_filters_first=n_filters_second,
                                       n_layers=n_layers_second)
        convnets_second.append(FlattenLayer(convnet))

    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool2 = ConcatLayer(convnets_second)
    convpool2 = ReshapeLayer(convpool2, ([0], n_timewin, get_output_shape(convnets_second[0])[1]))
    convpool2 = DimshuffleLayer(convpool2, (0, 2, 1))
    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    convpool2 = Conv1DLayer(convpool2, 64, 3)
    convpool2 = FlattenLayer(convpool2)

    # Merge 1D-Conv and LSTM outputs
    dense_input = ConcatLayer([convpool2, lstm_out])

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    for i in range(len(dense_num_unit)):
        convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5), num_units=dense_num_unit[i],
                              nonlinearity=lasagne.nonlinearities.rectify)
        if batch_norm_dense:
            convpool = batch_norm(convpool)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    if nb_classes == 1:
        nonlinearity = lasagne.nonlinearities.sigmoid
    else:
        nonlinearity = lasagne.nonlinearities.softmax
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                          num_units=nb_classes, nonlinearity=nonlinearity)
    return convpool


def network_custom_mlp(input_var, nb_classes, channel_size=20, n_colors=3, width=[256,256], drop_input=.2,
                       drop_hidden=.5):
    """
    Builds the mlp layer

    :param input_var: list of EEG features
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, n_colors, channel_size),
                                        input_var=input_var)
    network = FlattenLayer(network)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    for i in range(len(width)):
        network = lasagne.layers.DenseLayer(network, width[i], nonlinearity=lasagne.nonlinearities.rectify)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    network = lasagne.layers.DenseLayer(network, nb_classes, nonlinearity=lasagne.nonlinearities.sigmoid)
    return network


def network_custom_mlp_multi(input_var, nb_classes, channel_size=20, n_colors=3, width=256, drop_input=.2,
                             drop_hidden=.5, n_timewin=5):
    """
    Builds the mlp layer

    :param input_var: list of EEG features
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """
    networks = []
    for i in range(n_timewin):
        # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
        network = lasagne.layers.InputLayer(shape=(None, n_colors, channel_size),
                                            input_var=input_var[i])
        network = FlattenLayer(network)
        if drop_input:
            network = lasagne.layers.dropout(network, p=drop_input)
        # Hidden layers and dropout:
        for i in range(len(width)):
            network = lasagne.layers.DenseLayer(network, width[i], nonlinearity=lasagne.nonlinearities.rectify)
            if drop_hidden:
                network = lasagne.layers.dropout(network, p=drop_hidden)
        networks.append(FlattenLayer(network))
    networks = ConcatLayer(networks)
    # Output layer:
    networks = lasagne.layers.DenseLayer(networks, nb_classes, nonlinearity=lasagne.nonlinearities.sigmoid)
    return networks
