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
file_model = '../model/'

labelTypes = {
    "interictal": 0,
    "preictal": 1,
    "ictal": 2,
    "postictal": 3,
    "excluded": 4,
    "sph": 5
}

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
    def __init__(self):
        self.windowLength = 4
        self.overlap = 0.5
        self.pixelCount = [8, 8]
        self.genType = 'advanced'
        self.genMethod = ['RBF', 'CT2D'][1]
        self.edgeless = False
        self.featureTypes = range(14)


class TimingParams(object):
    def __init__(self):
        self.preictalLen = 30
        self.postictalLen = 10
        self.minPreictalLen = 15
        self.minInterictalLen = 30
        self.excludedLen = 120
        self.sphLen = 1


class TrainDataParams(object):
    def __init__(self):
        self.foldCount = 5
        self.seqWinCount = 1
        self.kFoldShuffle = True
        self.underSample = True
        self.dataRatio = [1, 1]
        self.subject = []
        self.evalExclued = False

class TrainParams(object):
    def __init__(self):
        self.numEpoch = 200
        self.batchSize = 200
        self.learnRate = 0.001
        self.l1 = 0.0
        self.l2 = 0.0
        self.onlineWeights = False
        self.saveModelParams = False
        self.preTrainedInit = False
        self.plotPrediction = False
        self.earlyStopping = {"active": True,
                              "monitor": "validErr",
                              "patience": 10,
                              "min_delta": 0.01}
        self.adaptiveL2 = {"active": True,
                           "monitor": "validAcc",
                           "treshold": [99, 97],
                           "epoch": 5,
                           "l2_step": 0.01}

class ModelParams(object):
    def __init__(self):
        self.modelName = None
        self.modelType = None
        self.pixelCount = [8, 8]
        self.seqWinCount = 1
        self.genImageType = 'advanced'
        self.networkParams = []
        self.networkSharedVars = []


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










