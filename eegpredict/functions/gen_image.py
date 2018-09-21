import scipy.io
import numpy as np
import itertools
from multiprocessing import Pool
from sklearn.preprocessing import scale
from scipy.interpolate import CloughTocher2DInterpolator, Rbf
from functions.math_utils import augment_EEG

def gen_images_basic(locs, features, dims=[4, 5], normalize=False):
    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] / nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])
    nSamples = features.shape[0]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, dims[0], dims[1]]))
    for c in range(n_colors):
        for i in range(locs.shape[0]):
            temp_interp[c][:, locs[i][0], locs[i][1]] = feat_array_temp[c][:, i]
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])], with_mean=True)
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)

class gen_images_engine(object):

    def __init__(self, nSamples, n_colors, locs, feat_array_temp, grid_x, grid_y, method):
        self.n_colors = n_colors
        self.locs = locs
        self.feat_array_temp = feat_array_temp
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.nSamples = nSamples
        self.method = method
    def __call__(self, c):
        temp_interp = []
        for i in range(self.nSamples):
            if self.method == 'CT2D':
                ip = CloughTocher2DInterpolator(self.locs, self.feat_array_temp[c][i, :], fill_value=np.nan)
                temp_interp.append(ip((self.grid_x, self.grid_y)))
            elif self.method == 'RBF':
                ip = Rbf(self.locs[:, 0], self.locs[:, 1], self.feat_array_temp[c][i, :],
                         function='cubic', smooth=0)
                temp_interp.append(ip(self.grid_x, self.grid_y))
        return temp_interp

def gen_images(locs, features, n_gridpoints, normalize=True, augment=False, pca=False, std_mult=0.1, n_components=2,
                   edgeless=False, multip = False, method='RBF'):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] == nElectrodes
    n_colors = features.shape[2]
    feat_array_temp = np.moveaxis(np.asarray(features), -1, 0)
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        feat_array_stack = []
        for c in range(n_colors):
            feat_array_stack.append(np.hstack((feat_array_temp[c], np.zeros((nSamples, 4)))))
        feat_array_temp = np.asarray(feat_array_stack)

    # Interpolating
    if multip:
        try:
            pool = Pool(8)  # on 8 processors
            engine = gen_images_engine(nSamples, n_colors, locs, feat_array_temp, grid_x, grid_y, method)
            temp_interp = pool.map(engine, range(n_colors))
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()
    else:
        for i in range(nSamples):
            for c in range(n_colors):
                # print('Interpolating {0}/{1}\r'.format(i+1, nSamples))
                if method == 'CT2D':
                    ip = CloughTocher2DInterpolator(locs, feat_array_temp[c][i, :], fill_value=np.nan)
                    temp_interp[c][i, :, :] = ip((grid_x, grid_y))
                elif method == 'RBF':
                    ip = Rbf(locs[:, 0], locs[:, 1], feat_array_temp[c][i, :],
                             function='cubic', smooth=0)
                    temp_interp[c][i, :, :] = ip(grid_x, grid_y)
    print('Interpolating...\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_norm = np.reshape(temp_interp[c], (-1))
            temp_norm[~np.isnan(temp_norm)] = \
                scale(temp_norm[~np.isnan(temp_norm)], with_mean=False)
            temp_interp[c] = np.reshape(temp_norm, np.asarray(temp_interp[c]).shape)
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]
