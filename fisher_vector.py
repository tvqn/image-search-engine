import numpy as np
import pdb
import os
import pickle

from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
import cv2
from tqdm import tqdm
import h5py

K = 128
N = 256 * 1000

def FisherVectorExtraction(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

def ToImageOpenCV(image, grayScale=False):
    npImage = np.array(image)
    if grayScale == True:
        return cv2.cvtColor(npImage, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(npImage, cv2.COLOR_RGB2BGR)

def SiftFeatureExtraction(pil_image, imageID, savePath):
    sift = cv2.xfeatures2d.SIFT_create()
    image = ToImageOpenCV(image=pil_image, grayScale=True)
    height, width = image.shape[:2]
    img_resize = cv2.resize(image, (int(0.5*width), int(0.5*height)))
    kp, des = sift.detectAndCompute(img_resize, None)
    
    with open(os.path.join(savePath, imageID + '.opencv.sift'), 'w') as fileSiftFeature:
        if des is None:
            fileSiftFeature.write(str(128) + '\n')
            fileSiftFeature.write(str(0) + '\n')
        elif len(des) > 0:
            fileSiftFeature.write(str(128) + '\n')
            fileSiftFeature.write(str(len(kp)) + '\n')
            for j in range(len(des)):
                locs_str = '0 0 0 0 0 '
                descs_str = " ".join([str(int(value)) for value in des[j]])
                all_strs = locs_str + descs_str
                fileSiftFeature.write(all_strs + '\n')
    
    return des
                
def TrainGaussianMixture(sample, n_components = 128, saveModelPath = None):
    gmm = GaussianMixture(n_components= n_components, covariance_type='diag')
    gmm.fit(sample)
    if saveModelPath is None:
        saveModelPath = os.path.join(os.getcwd(), "tmp", "gmm.pkl")
    SaveGaussianMixture(gmm, saveModelPath)
    
    return gmm

def SaveGaussianMixture(gmm, savePath):
    with open(savePath, 'wb') as file:
        pickle.dump(gmm, file)

def LoadGaussianMixture(savePath):
    loaded_gmm = GaussianMixture(n_components = K, covariance_type='diag')
    with open(savePath, 'rb') as file:
        loaded_gmm = pickle.load(file)
    return loaded_gmm

def LoadSiftFeatureStack(siftFeaturePath, randonSeed = 1024):
    # create stack vector of sift feature
    all_desc = []
    for i, item in enumerate(tqdm(os.listdir(siftFeaturePath), desc= "Load stack sift_feature")):
        hesaff_path = os.path.join(siftFeaturePath, item)
        hesaff_info = np.loadtxt(hesaff_path, skiprows=2)
        if hesaff_info.shape[0] == 0:
            continue
        elif hesaff_info.shape[0] > 0 and len(hesaff_info.shape) == 1:
            desc = hesaff_info[5:]
            all_desc.append(desc)
        elif hesaff_info.shape[0] > 0 and len(hesaff_info.shape) > 1:
            desc = hesaff_info[:, 5:]
            all_desc.append(desc)

    # make a big matrix with all image descriptors
    all_desc = np.sqrt(np.vstack(all_desc))

    # choose n_sample descriptors at random
    np.random.seed(randonSeed)
    sample_indices = np.random.choice(all_desc.shape[0], N)
    sample = all_desc[sample_indices]

    # until now sample was in uint8. Convert to float32
    return sample.astype('float32')

def LoadSiftFeature(siftFeaturePath):
    hesaff_info = np.loadtxt(siftFeaturePath, skiprows=2)
    if hesaff_info.shape[0] == 0:
        hesaff_info = np.zeros((1, 133), dtype = 'float32')
    elif hesaff_info.shape[0] > 0 and len(hesaff_info.shape) == 1:
        hesaff_info = hesaff_info.reshape([1, 133])

    image_desc = np.sqrt(hesaff_info[:, 5:])

    return image_desc

def ExtractionFeature(siftFeature, pca_transform = None, gmm = None):
    mean = gmm.means_
    image_desc = np.dot(image_desc - mean, pca_transform)
    image_desc = image_desc.astype(np.float32)

    return FisherVectorExtraction(sample, loaded_gmm)

def PCA_Transform(sample, save = False, savePath = None):
    # compute mean and covariance matrix for the PCA
    mean = sample.mean(axis = 0)
    sample = sample - mean
    cov = np.dot(sample.T, sample)

    # compute PCA matrix and keep only 64 dimensions
    eigvals, eigvecs = np.linalg.eig(cov)
    perm = eigvals.argsort()                   # sort by increasing eigenvalue
    pca_transform = eigvecs[:, perm[96:128]]   # eigenvectors for the 64 last eigenvalues
    if save == True:
        if savePath is None:
            savePath = os.path.join(os.getcwd(), "tmp", "pca_transform.gmm")
        np.save(savePath, pca_transform)
    # transform sample with PCA (note that numpy imposes line-vectors,
    # so we right-multiply the vectors)
    return np.dot(sample, pca_transform)

def BuildFisherVectorDict(config: dict):
    sampleStack = LoadSiftFeatureStack(siftFeaturePath= config['siftFeaturePath'])
    sampleStack = PCA_Transform(sample= sampleStack)

    gmm = GaussianMixture(n_components= config["n_components"], covariance_type='diag')
    if config["trainGMM"] == True:
        gmm = TrainGaussianMixture(
            sample= sample, 
            n_components = config['n_components'],
            saveModelPath = config['gmmSavePath'])
    else:
        gmm = LoadGaussianMixture(config['gmmSavePath'])

    features = []
    imgNames = []
    for item in tqdm(os.listdir(config['siftFeaturePath']), desc= "Load sift_feature"):
        sample = LoadSiftFeature(os.path.join(config["siftFeaturePath"], item))
        sample = PCA_Transform(sample= sample)
        fv = FisherVectorExtraction(sample, gmm)
        features.append(fv)
        imgNames.append(item.split(".")[0])
    # make one matrix with all FVs
    features = np.vstack(features)

    # power-normalization
    features = np.sign(features) * np.abs(features) ** 0.5

    h5f = h5py.File(config["featureSavePath"] , 'w')

    h5f['feats'] = features
    h5f['names'] = imgNames
    h5f.close()

def Query():
    pass

def main():
    # Short demo.

    xx, _ = make_classification(n_samples=N)
    xx_tr, xx_te = xx[: -100], xx[-100: ]

    gmm = GaussianMixture(n_components=K, covariance_type='diag')
    gmm.fit(xx_tr)

    fv = fisher_vector(xx_te, gmm)
    pdb.set_trace()

if __name__ == '__main__':
    # main()
    config = {
        "n_components" : 128,
        "n_sample" : 256 * 1000,
        "trainGMM": False,
        "siftFeaturePath": os.path.join(os.getcwd(), "sift_feature"),
        "gmmSavePath": os.path.join(os.getcwd(), "tmp", "gmm.pkl"),
        "featureSavePath": os.path.join(os.getcwd(), "tmp", 'fisher.h5')
    }

    BuildFisherVectorDict(config)

    # pdb.set_trace()