import numpy as np
import time
from itertools import product
from sklearn.neighbors import NearestNeighbors

def nearest_neighbor(points, error_margin, k, inputs, model=None):
    if model is None:
        model = NearestNeighbors(k, error_margin)
        model.fit(points)
    match_indices = model.kneighbors(inputs, k, return_distance=False).squeeze()
    k_matches = points[match_indices]
    print("Inputs:")
    print(inputs)
    print("Nearest Neighbors:")
    print(k_matches)
    return model, k_matches

if __name__ == '__main__':
    pixels = np.array(list(product(range(640),range(480)))) # get all pixels in (640, 480) image
    error_margin = 10 # pixel error margin
    k = 5 # 5 nearest neighbors
    X = np.random.randint(0, 640, (3, 1)) 
    Y = np.random.randint(0, 480, (3, 1))
    inputs = np.hstack((X, Y)) # get random array of pixels to get knn's for
    start = time.time()
    fitted_model, _ = nearest_neighbor(pixels, error_margin, k, inputs, model=None)
    print("First query time: %.4f" % (time.time() - start))

    X1 = np.random.randint(0, 640, (9, 1)) 
    Y1 = np.random.randint(0, 480, (9, 1))
    inputs = np.hstack((X1, Y1)) # get random array of pixels to get knn's for
    start = time.time()
    nearest_neighbor(pixels, error_margin, 100, inputs, model=fitted_model)
    print("Second query time: %.4f" % (time.time() - start))
