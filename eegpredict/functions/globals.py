import numpy as np

# Folder and File Definitions
# dataset_link='/Volumes/MacHDD/Dataset/physiobank/chbmit/'
dataset_link = 'D:/Dataset/physiobank/chbmit/'
file_records = '../data/refData/records.mat'
file_seizures = '../data/refData/seizureList.mat'
file_eeg_locs = '../data/refData/10_20_eeg_locs.mat'
file_eeg_locs2d = '../data/refData/locs2d_20c.mat'
file_feat_dataset = '../data/featData/'
file_img_dataset = '../data/imgData/'
file_lbl_dataset = '../data/lblData/'
file_test = '../test/'
file_test_db = '../test/test.db'
file_model = '../model/'

labelTypes = {
    "interictal": 0,
    "preictal": 1,
    "ictal": 2,
    "postictal": 3,
    "excluded": 4,
    "sph": 5
}

dbFeatGroups = ["featureParams", "timingParams", "trainDataParams", "trainParams", "modelParams"]
dbFeats = ['edgeless', 'featureTypes', 'pixelCount', 'overlap', 'genMethod', 'genImageType', 'windowLength',
           'preictalLen', 'sphLen', 'excludedLen', 'minInterictalLen', 'refractoryLen', 'minPreictalLen',
           'postictalLen',
           'dataRatio', 'underSample', 'kFoldShuffle', 'evalExcluded', 'seqWinCount', 'foldCount', 'subject',
           'batchSize', 'numEpoch', 'onlineWeights', 'l2', 'learnRate', 'l1', 'preTrainedInit', 'adaptiveL2',
           'earlyStopping',
           'modelName', 'modelType', 'seqWinCount'
           ]

# Deterministic Random
random_state = 1234
#random_state = None

# Global Classes
class EmptyClass(object):
    pass

class SubjectClass(object):
    def __init__(self):
        self.records = []
        self.all_labels = []
        self.interictal_parts = []
        self.preictal_parts = []


class FeatureParams(object):
    def __init__(self, windowLength=4, overlap=0.5, pixelCount=[8, 8], genImageType='advanced',
                 genMethod=['RBF', 'CT2D'][1], edgeless=False, featureTypes=["psd", "moment", "hjorth"]):
        self.windowLength = windowLength
        self.overlap = overlap
        self.pixelCount = pixelCount
        self.genImageType = genImageType
        self.genMethod = genMethod
        self.edgeless = edgeless
        self.featureTypes = self.init_feature_types(featureTypes)

    def init_feature_types(self, featureTypes):

        # Features
        feature_psd = range(0, 8)
        feature_moment = range(8, 12)
        feature_hjorth = range(12, 14)

        feature_eval = []
        if "psd" in featureTypes:
            feature_eval = np.hstack([feature_eval, feature_psd])

        if "moment" in featureTypes:
            feature_eval = np.hstack([feature_eval, feature_moment])

        if "hjorth" in featureTypes:
            feature_eval = np.hstack([feature_eval, feature_hjorth])

        return feature_eval.astype(int)



class TimingParams(object):
    def __init__(self, preictalLen=60, postictalLen=10, minPreictalLen=15, minInterictalLen=30, excludedLen=240,
                 refractoryLen=30, sphLen=1):
        self.preictalLen = preictalLen
        self.postictalLen = postictalLen
        self.minPreictalLen = minPreictalLen
        self.minInterictalLen = minInterictalLen
        self.excludedLen = excludedLen
        self.refractoryLen = refractoryLen
        self.sphLen = sphLen


class TrainDataParams(object):
    def __init__(self, foldCount = 5, seqWinCount=1, kFoldShuffle=True, underSample=True, dataRatio=[1, 1],subject=[],
                 evalExcluded=False):
        self.foldCount = foldCount
        self.seqWinCount = seqWinCount
        self.kFoldShuffle = kFoldShuffle
        self.underSample = underSample
        self.dataRatio = dataRatio
        self.subject = subject
        self.evalExcluded = evalExcluded

class TrainParams(object):
    def __init__(self, numEpoch=200, batchSize=200, learnRate=0.001, l1=0.0, l2=0.0, onlineWeights=False,
                 saveModelParams=False, preTrainedInit=False, plotPrediction=False, earlyStopping=True,
                 adaptiveL2=False):
        self.numEpoch = numEpoch
        self.batchSize = batchSize
        self.learnRate = learnRate
        self.l1 = l1
        self.l2 = l2
        self.onlineWeights = onlineWeights
        self.saveModelParams = saveModelParams
        self.preTrainedInit = preTrainedInit
        self.plotPrediction = plotPrediction
        self.earlyStopping = {"active": earlyStopping,
                              "monitor": "validErr",
                              "patience": 10,
                              "min_delta": 0.01}
        self.adaptiveL2 = {"active": adaptiveL2,
                           "monitor": "validAcc",
                           "treshold": [95, 90],
                           "epoch": 5,
                           "l2_step": 0.01}

class ModelParams(object):
    def __init__(self, modelName=None, modelType=None, pixelCount=[8, 8], seqWinCount=1, genImageType='advanced',
                 networkParams=[], networkSharedVars=[]):
        self.modelName = modelName
        self.modelType = modelType
        self.pixelCount = pixelCount
        self.seqWinCount = seqWinCount
        self.genImageType = genImageType
        self.networkParams = networkParams
        self.networkSharedVars = networkSharedVars


class Dataset(object):
    def __init__(self):
        self.data = []
        self.labels = []
        self.weights = []
        self.fold_pairs = []


class Data(object):
    def __init__(self):
        self.X = []
        self.Y = []
        self.W = []
        self.SWC = []


class TrainResult(object):
    def __init__(self):
        self.trainParams = []
        self.testParams = []
        self.netWeights = []

class EpochParams(object):
    def __init__(self):
        self.curEpoch = 0
        self.bestEpoch = 0
        self.epoch = []
        self.duration = []
        self.trainErr = []
        self.trainAcc = []
        self.validErr = []
        self.validAcc = []
        self.validPred = []
        self.trainValRatio = []
        self.netParamVal = []

class TestParams(object):
    def __init__(self):
        self.testErr = []
        self.testAcc = []
        self.testPred = []










