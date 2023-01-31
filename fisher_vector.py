import numpy as np
import pdb
import os
import pickle

from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture

K = 128
N = 256 * 1000

def fisher_vector(xx, gmm):
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

def SaveGaussianMixture(gmm, name, savePath):
    gmm_name = os.path.join(savePath, name)
    with open(gmm_name, 'wb') as file:
        pickle.dump(gmm, file)

def LoadGaussianMixture(name, savePath):
    gmm_name = os.path.join(savePath, name)
    loaded_gmm = GaussianMixture(n_components = K, covariance_type='diag')
    with open(gmm_name, 'rb') as file:
        loaded_gmm = pickle.load(file)
    return loaded_gmm

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
    K = 128
    N = 256 * 1000
    # create stack vector of sift feature
    siftFeaturePath = os.path.join(os.getcwd(), "sift_feature")
    all_desc = []
    for i, item in enumerate(os.listdir(siftFeaturePath)):
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
    print(all_desc.shape)

    # choose n_sample descriptors at random
    np.random.seed(1024)
    sample_indices = np.random.choice(all_desc.shape[0], N)
    sample = all_desc[sample_indices]

    # until now sample was in uint8. Convert to float32
    sample = sample.astype('float32')

    # compute mean and covariance matrix for the PCA
    mean = sample.mean(axis = 0)
    sample = sample - mean
    cov = np.dot(sample.T, sample)

    # compute PCA matrix and keep only 64 dimensions
    eigvals, eigvecs = np.linalg.eig(cov)
    perm = eigvals.argsort()                   # sort by increasing eigenvalue
    pca_transform = eigvecs[:, perm[96:128]]   # eigenvectors for the 64 last eigenvalues

    # transform sample with PCA (note that numpy imposes line-vectors,
    # so we right-multiply the vectors)
    sample = np.dot(sample, pca_transform)
    print(sample.shape)

    saveModelPath = os.path.join(os.getcwd(), "tmp")
    gmm = GaussianMixture(n_components=K, covariance_type='diag')
    # gmm.fit(sample)
    # SaveGaussianMixture(gmm, 'model.pkl', saveModelPath)
    # Serialize and save the model
    gmm_name = os.path.join(saveModelPath, )

    gmm = LoadGaussianMixture("model.pkl", saveModelPath)
    fv = fisher_vector(sample, gmm)
    pdb.set_trace()