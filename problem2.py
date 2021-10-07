'''
Author: Yuxuan Wu yuxuanw2@andrew.cmu.edu
Date: 2021-10-07 00:34:07
LastEditTime: 2021-10-07 18:32:51
LastEditors: Please set LastEditors
Description: Implementation of Forth Order Blind Indentification ICA
FilePath: /hw2/problem2.1.py
'''
import numpy as np
import librosa as lr
import os
from soundfile import write as sfwrite


def ica(X):
    '''
    Implementation of FOBI-ICA
    '''
    for i, x in enumerate(X):
        X[i] = x - np.mean(x)
    # Whiten the observations (decorrelating)
    C = np.dot(X, X.T)/X.shape[0]  # covariance matrix of X
    E, eigenvalues, _ = np.linalg.svd(C)  # eigenvalue and eigenvectors
    # construct diagonal matrix of eigenvalues
    D = np.zeros(E.shape)
    for i, row in enumerate(D):
        D[i][i] = 1/np.sqrt(eigenvalues[i])
    # get whitened
    XHat = np.dot(np.dot(D, E.T), X)

    # Then W in S=WXHat is unitary matrix
    # Calculate forth order correlation
    C = np.dot(np.linalg.norm(XHat, axis=0) * XHat, XHat.T)/XHat.shape[0]
    # Eigen Analysis
    E, eigenvalues, _ = np.linalg.svd(C)
    # Get source
    S = np.dot(E.T, XHat)

    return S

def mixingMatrix(X, S):
    A = np.dot(X, np.linalg.pinv(S))
    return A

if __name__ == '__main__':
    X = []
    dirname = "hw2_materials_f21/problem2"
    for i, filename in enumerate(os.listdir(dirname)):
        y, sr = lr.load(os.path.join(dirname, filename), 44100)
        X.append(y)
    X = np.array(X)
    S = ica(X)
    for i, source in enumerate(S):
        source /= np.max(source)
        sfwrite("source"+str(i)+".wav", source, sr)
        print("source"+str(i)+".wav Saved.")
    A = mixingMatrix(X, S)
    np.savetxt("mixing_matrix.csv", A)
    print("mixing_matrix.csv Saved.")
