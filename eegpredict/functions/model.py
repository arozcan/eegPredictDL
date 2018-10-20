import theano
import lasagne
import theano.tensor as T
from sklearn import svm
from lasagne import regularization
from functions.network_models import network_convpool_cnn, network_convpool_cnn_max, network_convpool_conv1d, \
    network_convpool_lstm, network_convpool_mix, network_custom_mlp, network_custom_mlp_multi, \
    network_convpool_lstm_hybrid, network_convpool_cnn3d, network_convpool_cnn_basic
from functions.globals import EmptyClass


def backup_network_params(model):

    if model.modelParams.modelType == 'neural':
        # network'un ilklendirilmis parametreleri restore edilmek uzere kaydedildi
        model.modelParams.networkParams = lasagne.layers.get_all_param_values(model.model)

        # backup shared vars
        model.modelParams.networkSharedVars = [x.get_value() for x in model.trainFn.get_shared()]


def restore_network_params(model):

    if model.modelParams.modelType == 'neural':
        # her epoch dongusunde model parametreleri restore ediliyor
        lasagne.layers.set_all_param_values(model.model, model.modelParams.networkParams)

        # restore shared vars
        for i, sharedVar in zip(range(len(model.modelParams.networkSharedVars)), model.modelParams.networkSharedVars):
            model.trainFn.get_shared()[i].set_value(sharedVar)


def get_network_params(model):

    if model.modelParams.modelType == 'neural':
        # network'un ilklendirilmis parametreleri restore edilmek uzere kaydedildi
        return lasagne.layers.get_all_param_values(model.model)


def set_network_params(model, params):

    if model.modelParams.modelType == 'neural':
        # her epoch dongusunde model parametreleri restore ediliyor
        lasagne.layers.set_all_param_values(model.model, params)



def create_neural_network_model(modelParams, featureParams, inputVar, numClassUnit):

    # Create model with parameters
    if modelParams.modelName == 'model_custom_mlp':
        inputVar = T.tensor3('inputs')
        network = network_custom_mlp(inputVar,
                                     numClassUnit,
                                     n_colors=len(featureParams.featureTypes),
                                     channel_size=modelParams.pixelCount,
                                     width=modelParams.denseNumUnit,
                                     drop_input=modelParams.dropoutInput,
                                     drop_hidden=modelParams.dropoutDense)

    elif modelParams.modelName == 'model_custom_mlp_multi':
        inputVar = T.tensor4('inputs')
        network = network_custom_mlp_multi(inputVar,
                                           numClassUnit,
                                           n_colors=len(featureParams.featureTypes),
                                           channel_size=modelParams.pixelCount,
                                           width=modelParams.denseNumUnit,
                                           drop_input=modelParams.dropoutInput,
                                           drop_hidden=modelParams.dropoutDense,
                                           n_timewin=modelParams.seqWinCount)

    elif modelParams.modelName == 'model_cnn':
        inputVar = T.tensor4('inputs')
        network = network_convpool_cnn(inputVar,
                                       numClassUnit,
                                       n_colors=len(featureParams.featureTypes),
                                       imsize=modelParams.pixelCount,
                                       n_layers=modelParams.nLayers,
                                       n_filters_first=modelParams.nFiltersFirst,
                                       dense_num_unit=modelParams.denseNumUnit,
                                       batch_norm_conv=modelParams.batchNormConv)

    elif modelParams.modelName == 'model_cnn_basic':
        inputVar = T.tensor4('inputs')
        network = network_convpool_cnn_basic(inputVar,
                                             numClassUnit,
                                             n_colors=len(featureParams.featureTypes),
                                             imsize=modelParams.pixelCount,
                                             n_layers=modelParams.nLayers,
                                             n_filters_first=modelParams.nFiltersFirst,
                                             dense_num_unit=modelParams.denseNumUnit,
                                             batch_norm_conv=modelParams.batchNormConv)

    elif modelParams.modelName == 'model_cnn_max':
        network = network_convpool_cnn_max(inputVar,
                                           numClassUnit,
                                           n_colors=len(featureParams.featureTypes),
                                           imsize=modelParams.pixelCount,
                                           n_timewin=modelParams.seqWinCount,
                                           n_layers=modelParams.nLayers,
                                           n_filters_first=modelParams.nFiltersFirst,
                                           dense_num_unit=modelParams.denseNumUnit,
                                           batch_norm_conv=modelParams.batchNormConv)

    elif modelParams.modelName == 'model_cnn_conv1d':
        network = network_convpool_conv1d(inputVar,
                                          numClassUnit,
                                          n_colors=len(featureParams.featureTypes),
                                          imsize=modelParams.pixelCount,
                                          n_timewin=modelParams.seqWinCount,
                                          n_layers=modelParams.nLayers,
                                          n_filters_first=modelParams.nFiltersFirst,
                                          dense_num_unit=modelParams.denseNumUnit,
                                          batch_norm_conv=modelParams.batchNormConv)

    elif modelParams.modelName == 'model_cnn_lstm':
        network = network_convpool_lstm(inputVar,
                                        numClassUnit,
                                        n_colors=len(featureParams.featureTypes),
                                        imsize=modelParams.pixelCount,
                                        n_timewin=modelParams.seqWinCount,
                                        n_layers=modelParams.nLayers,
                                        n_filters_first=modelParams.nFiltersFirst,
                                        dense_num_unit=modelParams.denseNumUnit,
                                        batch_norm_conv=modelParams.batchNormConv)

    elif modelParams.modelName == 'model_cnn_mix':
        network = network_convpool_mix(inputVar,
                                       numClassUnit,
                                       n_colors=len(featureParams.featureTypes),
                                       imsize=modelParams.pixelCount,
                                       n_timewin=modelParams.seqWinCount,
                                       n_layers=modelParams.nLayers,
                                       n_filters_first=modelParams.nFiltersFirst,
                                       dense_num_unit=modelParams.denseNumUnit,
                                       batch_norm_conv=modelParams.batchNormConv)

    elif modelParams.modelName == 'model_cnn_lstm_hybrid':
        network = network_convpool_lstm_hybrid(inputVar,
                                               numClassUnit,
                                               n_colors=len(featureParams.featureTypes),
                                               imsize=modelParams.pixelCount,
                                               n_timewin=modelParams.seqWinCount,
                                               n_layers=modelParams.nLayers,
                                               n_filters_first=modelParams.nFiltersFirst,
                                               dense_num_unit=modelParams.denseNumUnit,
                                               batch_norm_conv=modelParams.batchNormConv)

    elif modelParams.modelName == 'model_cnn3d' or modelParams.modelName == 'model_cnn3d_new':
        network = network_convpool_cnn3d(inputVar,
                                         numClassUnit,
                                         n_colors=len(featureParams.featureTypes),
                                         n_timewin=modelParams.seqWinCount,
                                         imsize=modelParams.pixelCount,
                                         n_layers=modelParams.nLayers,
                                         n_filters_first=modelParams.nFiltersFirst,
                                         dense_num_unit=modelParams.denseNumUnit,
                                         batch_norm_conv=modelParams.batchNormConv,
                                         pool_size=modelParams.poolSize,
                                         filter_size=modelParams.filterSize,
                                         filter_factor=modelParams.filterFactor,
                                         dropout_dense=modelParams.dropoutDense)
    else:
        raise ValueError("Model not supported")
    return network, inputVar

def create_neural_network(modelParams, featureParams, trainParams, trainDataParams):

    # Number of class unit
    numClassUnit = 1
    # Define Theano variables
    inputVar = T.TensorType('floatX', ((False,) * 5))()
    targetVar = T.ivector('targetVar')
    trainWeights = T.dvector('trainWeights')
    l2Adaptive = T.dscalar('l2Adaptive')

    # create neural network model with given parameters
    network, inputVar = create_neural_network_model(modelParams, featureParams, inputVar, numClassUnit)

    # Train icin loss fonksiyonu tanimlaniyor
    prediction = lasagne.layers.get_output(network)
    if numClassUnit == 1:
        trainLoss = lasagne.objectives.binary_crossentropy(prediction, targetVar)
        trainAcc = lasagne.objectives.binary_accuracy(prediction, targetVar, threshold=0.5)
        trainAcc = T.mean(trainAcc, dtype=theano.config.floatX)
    else:
        trainLoss = lasagne.objectives.categorical_crossentropy(prediction, targetVar)
        trainAcc = T.mean(T.eq(T.argmax(prediction, axis=1), targetVar), dtype=theano.config.floatX)

    # Loss fonksiyonun veriseti underSample yapilmadiysa agirlikli olarak, yapildiysa basit olarak ortalamasi aliniyor
    if trainDataParams.underSample == False:
        # ornek oranina gore agirliklandirma
        # interictal=0, preictal=1
        weights_per_label = theano.shared(lasagne.utils.floatX(trainDataParams.dataRatio))
        weights = weights_per_label[targetVar]
        trainLoss = lasagne.objectives.aggregate(trainLoss, weights=weights)
    else:
        if trainParams.onlineWeights:
            trainLoss = lasagne.objectives.aggregate(trainLoss, weights=trainWeights)
        else:
            trainLoss = trainLoss.mean()

    # regularization
    if trainParams.l1:
        trainLoss += regularization.regularize_network_params(network, regularization.l1) * trainParams.l1
    if trainParams.adaptiveL2["active"]:
        trainLoss += regularization.regularize_network_params(network, regularization.l2) * l2Adaptive
    elif trainParams.l2:
        trainLoss += regularization.regularize_network_params(network, regularization.l2) * trainParams.l2

    # Parametreleri update edecek foknsiyon
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(trainLoss, params, learning_rate=trainParams.learnRate)

    # Validation ve test icin loss fonksiyonu tanimlaniyor
    testPrediction = lasagne.layers.get_output(network, deterministic=True)
    if numClassUnit == 1:
        testLoss = lasagne.objectives.binary_crossentropy(testPrediction, targetVar)
        testLoss = testLoss.mean()
        testAcc = lasagne.objectives.binary_accuracy(testPrediction, targetVar, threshold=0.5)
        testAcc = T.mean(testAcc, dtype=theano.config.floatX)
    else:
        testLoss = lasagne.objectives.categorical_crossentropy(testPrediction, targetVar)
        testLoss = testLoss.mean()
        testAcc = T.mean(T.eq(T.argmax(testPrediction, axis=1), targetVar), dtype=theano.config.floatX)


    # Train fonksiyonu compile ediliyor
    trainFn = theano.function([inputVar, targetVar, trainWeights, l2Adaptive], [trainLoss, trainAcc], updates=updates,
                              on_unused_input='ignore')
    # Validation fonksiyonu compile ediliyor
    valFn = theano.function([inputVar, targetVar], [testLoss, testAcc, testPrediction])
    # Test fonksiyonu compile ediliyor
    testFn = theano.function([inputVar, targetVar], [testLoss, testAcc, testPrediction])

    return network, trainFn, valFn, testFn


def set_model_functions(mdl):

    # set models functions
    mdl.model.get_weights = get_network_params
    mdl.model.set_weights = set_network_params
    mdl.model.restore_model = restore_network_params
    mdl.model.backup_model = backup_network_params

    # backup model's initialized parameters
    mdl.model.backup_model(mdl)


def create_model(modelParams, featureParams, trainParams, trainDataParams):
    mdl = EmptyClass()
    mdl.modelParams = modelParams
    mdl.trainParams = trainParams

    if modelParams.modelType == 'svm':
        # create svm model
        mdl.model = svm.SVC(kernel=modelParams.kernel, class_weight='balanced', probability=False)

    elif modelParams.modelType == 'neural':
        # create neural network model
        mdl.model, mdl.trainFn, mdl.valFn, mdl.testFn = create_neural_network(modelParams, featureParams,
                                                                               trainParams, trainDataParams)
    else:
        exit('Unsupported model!')

    set_model_functions(mdl)

    return mdl
