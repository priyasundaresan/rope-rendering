import numpy as np
import os
import cv2
from scipy.interpolate import interp1d

def noisy(noise_typ,image,mask=None):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        if mask is not None:
            noisy[np.where((mask == [0]))] = 0
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2.0 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

def generate_perlin_noise_2d(shape, res=(12, 16)):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

if __name__ == '__main__':
    os.system('rm -rf ./images_noisy')
    os.makedirs('./images_noisy')
    a, b = 207, 120 # NOTE: a hack!! this is the range of values on the rope in sim (roughly)
    c, d = 137, 156 # NOTE: a hack!! this is the range of values on the rope in real (roughly)
    m = interp1d([a,b],[c,d]) 
    if not os.path.exists("./images_noisy"):
        os.makedirs('./images_noisy')
    else:
        os.system("rm -rf ./images_noisy")
        os.makedirs("./images_noisy")
    for f in os.listdir('./images'):
        if 'png' in f:
            print(f)
            img = cv2.imread('./images/' + f, 0).copy()
            img[np.where((img < [5]))] = [0]            
            mask = cv2.imread('./image_masks/' + f.replace('rgb', 'visible_mask'), 0).copy()
            #img = cv2.bitwise_not(img, mask=mask) # the depth in real is opposite sim (closer = darker), adjusts for this
            edges = cv2.Canny(img,0,255)
            edges = cv2.dilate(edges,(5, 5),iterations = 2)
            perlin_random = generate_perlin_noise_2d(img.shape)
            random = np.random.random(img.shape)
            img[np.where((edges != [0]) & (perlin_random > [0.0]) & (random > [0.4]))] = [0]
            perlin_random = generate_perlin_noise_2d(img.shape)
            perlin_random[np.where((img == [0]))] = [0]
            #img += cv2.normalize(perlin_random, None, -10, 10, cv2.NORM_MINMAX).astype('uint8')
            for val in range(b, a+1): #TODO: there should be a way to make this run in parallel
                img[np.where((img == [val]))] = [m(val)]
            if np.random.uniform() < 0.5:
                mode = 'gauss'
            else:
                mode = 'poisson'
            img_noise = noisy(mode, img, mask=mask)
            #img_noise = img 
            cv2.imwrite('./images_noisy/' + f, img_noise)
            
