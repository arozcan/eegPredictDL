import time
import itertools as it
import numpy as np
from functions import globals


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def dashed_line():
    return "\n----------------------------------------------------------------------------------"


def find_ranges(lst, val=0):
    groups = ((k, tuple(g)) for k, g in it.groupby(enumerate(lst), lambda x: x[-1]))
    repeated = (idx_g for k, idx_g in groups if k == val)
    return list([sub[0][0], sub[-1][0]+1] for sub in repeated)


def print_params(featureParams, timingParams, trainDataParams, trainParams, modelParams):

    params = {"featureParams": featureParams,
              "timingParams": timingParams,
              "trainDataParams": trainDataParams,
              "trainParams": trainParams,
              "modelParams": modelParams}

    keys = []
    for groups in globals.dbFeatGroups:
        params_list = "\t" + groups + ":\t"
        group_keys = params.get(groups).__dict__.keys()
        for key, idx in zip(group_keys, range(len(group_keys))):
            if (key not in keys) and (key in globals.dbFeats):
                keys.append(key)
                params_list += str(key) + "="
                val = params.get(groups).__dict__.get(key)
                if isinstance(val, basestring):
                    params_list += "'" + str(val) + "'"
                elif isinstance(val, np.ndarray) or isinstance(val, list):
                    params_list += "'" + str(val).replace(" ", "") + "'"
                elif isinstance(val, bool) or isinstance(val, int):
                    params_list += str(int(val))
                elif isinstance(val, dict):
                    params_list += str(int(val["active"]))
                elif isinstance(val, float):
                    params_list += str(float(val))
                params_list += " , "
        print(params_list)


def merge_dicts(dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result