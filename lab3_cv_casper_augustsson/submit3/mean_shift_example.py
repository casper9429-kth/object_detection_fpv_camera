import sys
import math
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from lab3 import kmeans_segm
from Functions import showgrey, mean_segments, overlay_bounds
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance_matrix

def mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations):

    print('Find colour channels with K-means...')
    K = 16 # number of channels
    [ segm, centers ] = kmeans_segm(I, K, 20, 4321)
    ( height, width, depth ) = np.shape(I)
    idx = np.reshape(segm, (height, width))
    mapsw = np.zeros((height, width, K))
    mapsx = np.zeros((height, width, K))
    mapsy = np.zeros((height, width, K))
    [X, Y] = np.meshgrid(range(width), range(height))
    for k in range(K):
        mapsw[:,:,k] = (idx == k).astype(float)
        mapsx[:,:,k] = gaussian_filter(mapsw[:,:,k]*X, spatial_bandwidth, mode='nearest')
        mapsy[:,:,k] = gaussian_filter(mapsw[:,:,k]*Y, spatial_bandwidth, mode='nearest')
        mapsw[:,:,k] = gaussian_filter(mapsw[:,:,k],   spatial_bandwidth, mode='nearest')
    mapsw = np.reshape(mapsw, (-1, K)) + 1e-6
    mapsx = np.reshape(mapsx, (-1, K))
    mapsy = np.reshape(mapsy, (-1, K))

    print('Search for high density points...')
    constC = -0.5/(colour_bandwidth**2)
    x = np.reshape(X, (width*height, ))
    y = np.reshape(Y, (width*height, ))
    Ic = np.reshape(I, (width*height, 3))
    wei = np.exp(constC*(distance_matrix(Ic, centers)**2))
    for l in range(num_iterations):
        p = (np.round(y)*width + np.round(x)).astype(int)
        ww = mapsw[p,:] * wei
        w = np.sum(ww, axis=1)
        u = (np.matmul(ww, centers).T / w).T
        x = ((np.sum(mapsx[p,:] * wei, axis=1)).T / w).T
        y = ((np.sum(mapsy[p,:] * wei, axis=1)).T / w).T
        wei = (ww.T / w).T
        x = np.maximum(np.minimum(x, width-1), 0);
        y = np.maximum(np.minimum(y, height-1), 0);

    print('Assign high density points to pixels...')
    XY = np.stack((x, y))
    thr = 4.0
    val = 0
    mask = np.zeros((height*width, 1), dtype=np.short)
    for y in range(height):
        for x in range(width):
            p = y*width + x
            if mask[p] == 0:
                stack = [ p ]
                val = val + 1
                mask[p] = val
                while len(stack) > 0:
                    p0 = stack[-1]
                    xy = XY[:, p0]
                    y0 = int(p0/width)
                    x0 = p0 - y0*width
                    stack = stack[:-1]
                    pn = p0 + 1
                    if x0<width-1 and mask[pn]==0 and (np.sum((xy - XY[:, pn])**2)<thr):
                        stack = stack + [ pn ]
                        mask[pn] = val
                    pn = p0 - 1
                    if x0>0 and mask[pn]==0 and (np.sum((xy - XY[:, pn])**2)<thr):
                        stack = stack + [ pn ]
                        mask[pn] = val
                    pn = p0 + width
                    if y0<height-1 and mask[pn]==0 and (np.sum((xy - XY[:, pn])**2)<thr):
                        stack = stack + [ pn ]
                        mask[pn] = val
                    pn = p0 - width
                    if y0>0 and mask[pn]==0 and (np.sum((xy - XY[:, pn])**2)<thr):
                        stack = stack + [ pn ]
                        mask[pn] = val
    segm = np.reshape(mask, (height, width))
    return segm


def mean_shift_example():
    scale_factor = 0.5         # image downscale factor
    image_sigma = 1.0          # image preblurring scale
    spatial_bandwidth = 10.0   # spatial bandwidth
    colour_bandwidth = 10.0    # colour bandwidth
    num_iterations = 10        # number of mean-shift iterations
    
    # tiger 1 5,50
    # tiger 2 5,40
    # orange 5,10
    
    
    img = Image.open('Images-jpg/orange.jpg')
    img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))
     
    h = ImageFilter.GaussianBlur(image_sigma)
    I = np.asarray(img.filter(ImageFilter.GaussianBlur(image_sigma))).astype(np.float32)
    
    segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations)
    Inew = mean_segments(img, segm)
    if True:
        Inew = overlay_bounds(img, segm)

    img = Image.fromarray(Inew.astype(np.ubyte))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img.save('result/meanshift.png')

if __name__ == '__main__':
    sys.exit(mean_shift_example())




