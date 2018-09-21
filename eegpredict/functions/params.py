# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from functions.globals import ModelParams

def init_model_parameters(modelName, pixelCount):
    modelParameters = ModelParams()
    modelParameters.modelName = modelName
    if modelName == 'model_svm':

        modelParameters.modelType = "svm"

        if pixelCount == 20:
            # goruntu boyutu
            modelParameters.pixelCount = pixelCount
        else:
            exit('Model does not support given pixelCount!')

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 1

        # Imge olusturma yontemi
        modelParameters.genImageType = 'none'

        # svm kernel type
        modelParameters.kernel = 'rbf'


    elif modelName == 'model_custom_mlp':

        modelParameters.modelType = "neural"

        if pixelCount == 20:
            # goruntu boyutu
            modelParameters.pixelCount = pixelCount
        else:
            exit('Model does not support given pixelCount!')

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 1

        # Imge olusturma yontemi
        modelParameters.genImageType = 'none'

        # Tam Bagli Katman Hucre Sayisi
        modelParameters.denseNumUnit = [512, 512, 512]

        # Giriste dropout
        modelParameters.dropoutInput = 0.0

        # Tam bagli katmanda dropout
        modelParameters.dropoutDense = 0.5

    elif modelName == 'model_custom_mlp_multi':

        modelParameters.modelType = "neural"

        if pixelCount == 20:
            # goruntu boyutu
            modelParameters.pixelCount = pixelCount
        else:
            exit('Model does not support given pixelCount!')

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 8

        # Imge olusturma yontemi
        modelParameters.genImageType = 'none'

        # Tam Bagli Katman Hucre Sayisi
        modelParameters.denseNumUnit = [512, 512, 512]

        # Giriste dropout
        modelParameters.dropoutInput = 0.2

        # Tam bagli katmanda dropout
        modelParameters.dropoutDense = 0.5

    elif modelName == 'model_cnn_basic':

        modelParameters.modelType = "neural"

        if pixelCount == [4, 5]:
            # goruntu boyutu
            modelParameters.pixelCount = pixelCount
        else:
            exit('Model does not support given pixelCount!')

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 1

        # Imge olusturma yontemi
        modelParameters.genImageType = 'basic'

        # CNN katman Sayisi
        modelParameters.nLayers = [3]

        # Ilk filtre boyutu
        modelParameters.nFiltersFirst = 32

        # Tam Bagli Katman Hucre Sayisi
        modelParameters.denseNumUnit = [128]

    elif modelName == 'model_cnn':

        modelParameters.modelType = "neural"

        # goruntu boyutu
        modelParameters.pixelCount = pixelCount

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 1

        # Imge olusturma yontemi
        modelParameters.genImageType = 'advanced'

        if modelParameters.pixelCount == [8, 8]:

            # CNN Katman Sayisi
            modelParameters.nLayers = [2, 1]

            # Ilk filtre boyutu
            modelParameters.nFiltersFirst = 16

            # Tam Bagli Katman Hucre Sayisi
            modelParameters.denseNumUnit = [128, 128]

            # CNN icinde batch normalizasyon
            modelParameters.batchNormConv = False

        elif modelParameters.pixelCount == [16, 16]:

            # CNN Katman Sayisi
            modelParameters.nLayers = [3, 2, 1]

            # Ilk filtre boyutu
            modelParameters.nFiltersFirst = 32

            # Tam Bagli Katman Hucre Sayisi
            modelParameters.denseNumUnit = [512, 512]

            # CNN icinde batch normalizasyon
            modelParameters.batchNormConv = False

        else:
            exit('Model does not support given pixelCount!')


    elif modelName == 'model_cnn_max':

        modelParameters.modelType = "neural"

        # goruntu boyutu
        modelParameters.pixelCount = pixelCount

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 8

        # Imge olusturma yontemi
        modelParameters.genImageType = 'advanced'

        if modelParameters.pixelCount == [8, 8]:

            # CNN Katman Sayisi
            modelParameters.nLayers = [2, 1]

            # Ilk filtre boyutu
            modelParameters.nFiltersFirst = 16

            # Tam Bagli Katman Hucre Sayisi
            modelParameters.denseNumUnit = [128, 128]

            # CNN icinde batch normalizasyon
            modelParameters.batchNormConv = False

        else:
            exit('Model does not support given pixelCount!')


    elif modelName == 'model_cnn_conv1d':

        modelParameters.modelType = "neural"

        # goruntu boyutu
        modelParameters.pixelCount = pixelCount

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 8

        # Imge olusturma yontemi
        modelParameters.genImageType = 'advanced'

        if modelParameters.pixelCount == [8, 8]:

            # CNN Katman Sayisi
            modelParameters.nLayers = [2, 1]

            # Ilk filtre boyutu
            modelParameters.nFiltersFirst = 16

            # Tam Bagli Katman Hucre Sayisi
            modelParameters.denseNumUnit = [256, 256]

            # CNN icinde batch normalizasyon
            modelParameters.batchNormConv = False

        else:
            exit('Model does not support given pixelCount!')

    elif modelName == 'model_cnn_lstm':

        modelParameters.modelType = "neural"

        # goruntu boyutu
        # [16, 16], [8, 8]
        modelParameters.pixelCount = pixelCount

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 8

        # Imge olusturma yontemi
        modelParameters.genImageType = 'advanced'

        if modelParameters.pixelCount == [8, 8]:

            # CNN Katman Sayisi
            modelParameters.nLayers = [2, 1]

            # Ilk filtre boyutu
            modelParameters.nFiltersFirst = 16

            # Tam Bagli Katman Hucre Sayisi
            modelParameters.denseNumUnit = [256, 256]

            # CNN icinde batch normalizasyon
            modelParameters.batchNormConv = False

        else:
            exit('Model does not support given pixelCount!')

    elif modelName == 'model_cnn_mix':

        modelParameters.modelType = "neural"

        # goruntu boyutu
        # [16, 16], [8, 8]
        modelParameters.pixelCount = pixelCount

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 8

        # Imge olusturma yontemi
        modelParameters.genImageType = 'advanced'

        if modelParameters.pixelCount == [8, 8]:

            # CNN Katman Sayisi
            modelParameters.nLayers = [2, 1]

            # Ilk filtre boyutu
            modelParameters.nFiltersFirst = 16

            # Tam Bagli Katman Hucre Sayisi
            modelParameters.denseNumUnit = [256, 256]

            # CNN icinde batch normalizasyon
            modelParameters.batchNormConv = False

        else:
            exit('Model does not support given pixelCount!')

    elif modelName == 'model_cnn_lstm_hybrid':

        modelParameters.modelType = "neural"

        # goruntu boyutu
        # [16, 16], [8, 8]
        modelParameters.pixelCount = pixelCount

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 8

        # Imge olusturma yontemi
        modelParameters.genImageType = 'advanced'

        if modelParameters.pixelCount == [8, 8]:

            # CNN Katman Sayisi
            modelParameters.nLayers = [2, 1]

            # Ilk filtre boyutu
            modelParameters.nFiltersFirst = 16

            # Tam Bagli Katman Hucre Sayisi
            modelParameters.denseNumUnit = [256, 256]

            # CNN icinde batch normalizasyon
            modelParameters.batchNormConv = False

        else:
            exit('Model does not support given pixelCount!')

    elif modelName == 'model_cnn3d':

        modelParameters.modelType = "neural"

        # goruntu boyutu
        # [16, 16], [8, 8]
        modelParameters.pixelCount = pixelCount

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 16

        # Imge olusturma yontemi
        modelParameters.genImageType = 'advanced'

        if modelParameters.pixelCount == [8, 8]:

            # CNN Katman Sayisi
            modelParameters.nLayers = [1, 2, 1]

            # Ilk filtre boyutu
            modelParameters.nFiltersFirst = 8

            # Tam Bagli Katman Hucre Sayisi
            modelParameters.denseNumUnit = [256, 256]

            # CNN icinde batch normalizasyon
            modelParameters.batchNormConv = False

            # CNN maxpool boyutu
            modelParameters.poolSize = [(2,1,1),(2,2,2),(2,2,2)]

            # CNN filter boyutu
            modelParameters.filterSize = [(3, 1, 1), (3, 3, 3), (3, 3, 3)]

            # CNN filter sayisi carpani
            modelParameters.filterFactor = 2

            # Tam bagli katmanda dropout
            modelParameters.dropoutDense = True

    elif modelName == 'model_cnn3d_new':

        modelParameters.modelType = "neural"

        # goruntu boyutu
        # [16, 16], [8, 8]
        modelParameters.pixelCount = pixelCount

        # ardisil pencere sayisi
        modelParameters.seqWinCount = 8

        # Imge olusturma yontemi
        modelParameters.genImageType = 'advanced'

        if modelParameters.pixelCount == [8, 8]:
            # CNN Katman Sayisi
            modelParameters.nLayers = [2, 1]

            # Ilk filtre boyutu
            modelParameters.nFiltersFirst = 16

            # Tam Bagli Katman Hucre Sayisi
            modelParameters.denseNumUnit = [256, 256]

            # CNN icinde batch normalizasyon
            modelParameters.batchNormConv = False

            # CNN maxpool boyutu
            modelParameters.poolSize = [(2, 2, 2), (2, 2, 2)]

            # CNN filter boyutu
            modelParameters.filterSize = [(3, 3, 3), (3, 3, 3)]

            # CNN filter sayisi carpani
            modelParameters.filterFactor = 2

            # Tam bagli katmanda dropout
            modelParameters.dropoutDense = True

        else:
            exit('Model does not support given pixelCount!')

    return modelParameters


def init_feature_types(psd=True, moment=True, hjorth=True):

    # Features
    feature_psd = range(0, 8)
    feature_moment = range(8, 12)
    feature_hjorth = range(12, 14)

    feature_eval = []
    if psd:
        feature_eval = np.hstack([feature_eval, feature_psd])

    if moment:
        feature_eval = np.hstack([feature_eval, feature_moment])

    if hjorth:
        feature_eval = np.hstack([feature_eval, feature_hjorth])

    feature_eval = feature_eval.astype(int)

    return feature_eval
