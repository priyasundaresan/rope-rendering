import numpy as np
from sklearn import cluster
import cv2
import matplotlib.pyplot as plt
from skimage import color
import scipy


def quantize(raster, n_colors):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))
    model = cluster.KMeans(n_clusters=n_colors)
    labels = model.fit_predict(reshaped_raster)
    palette = model.cluster_centers_
    np.save('rope_centroids.npy', palette[:, [1,2]])
    #quantized_raster = np.reshape(palette[labels], (width, height, palette.shape[1]))
    #return quantized_raster, palette
    return palette[:, [1,2]]


if __name__ == '__main__':
    result = None
    #for i in range(len(os.listdir("images"))):
    for i in range(1):
        rgb_raster = scipy.misc.imread("images/00000%d.jpg" % i)
        lab_raster = color.rgb2lab(rgb_raster)
        #scipy.misc.imsave("raster.jpg", lab_raster)
        quantized = quantize(lab_raster, 16)
        if result is not None:
            result = 0.5*(result + quantized)
        else:
            result = quantized
        print(result)
        #scipy.misc.imsave("quantized.jpg", quantized)
    
