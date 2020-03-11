import cv2
import numpy as np
import scipy.stats as st
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pprint

def normalize(x):
    return x/x.sum()

def gauss_2d(width, height, sigma, u, v, mask=None, normalize=False):
    mu_x = u
    mu_y = v
    X,Y=np.meshgrid(np.linspace(0,width,width),np.linspace(0,height,height))
    G=np.exp(-((X-mu_x)**2+(Y-mu_y)**2)/(2.0*sigma**2))
    if mask is not None:
        G[np.where((mask == [0,0,0]).all(axis=2))] = 0.0
    if normalize:
        return normalize(G)
    return G

def gauss_2d_batch(width, height, sigma, U, V, mask=None, normalize=False):
    U = np.expand_dims(U, axis=2) 
    V = np.expand_dims(V, axis=2) 
    X,Y=np.meshgrid(np.linspace(0,width-1,width),np.linspace(0,height-1,height))
    print(X, Y)
    X = np.repeat(X[np.newaxis, :, :], U.shape[0], axis=0)
    Y = np.repeat(Y[np.newaxis, :, :], V.shape[0], axis=0)
    G=np.exp(-((X-U)**2+(Y-V)**2)/(2.0*sigma**2))
    if mask is not None:
        mask = np.repeat(mask[np.newaxis, :, :], U.shape[0], axis=0)
        G[np.where((mask == [0]))] = 0.0
    if normalize:
        return normalize(G)
    return G

def bimodal_gauss(G1, G2, normalize=False):
    bimodal = np.maximum(G1, G2)
    if normalize:
        return normalize(bimodal)
    return bimodal
    
def show_distribution(G):
    cv2.imshow("result", G)
    cv2.waitKey(0)

if __name__ == '__main__':
    sigma = 20
    u1, v1 = 214, 350
    u2, v2 = 357, 180
    mask = None
    g1 = gauss_2d(640, 480, sigma, u1, v1, mask=mask)
    #g2 = gauss_2d(640, 480, sigma, u2, v2, mask=mask)
    #bimodal = bimodal_gauss(g1, g2)
    show_distribution(g1)
    #show_distribution(g1)
    #U = np.random.randint(0, 640, (4, 1)) 
    #V = np.random.randint(0, 480, (4, 1))
    #g2 = gauss_2d_batch(640, 480, sigma, U, V, mask=mask)
    #bimodal = bimodal_gauss(g1, g2)
    #for i in range(bimodal.shape[0]):
    #    show_distribution(bimodal[i])
