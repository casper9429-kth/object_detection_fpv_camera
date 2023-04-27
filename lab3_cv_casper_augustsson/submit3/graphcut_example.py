import sys
import math
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from lab3 import kmeans_segm, mixture_prob
from Functions import showgrey, mean_segments, overlay_bounds
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance_matrix
from scipy.signal import convolve2d

#   Performing the continuous max-flow algorithm to solve the 
#   continuous min-cut problem in 2D
#    
#   Usage: [u, erriter, i, timet] = CMF_Cut;
#
#   Inputs: there is no input since all data and parameters can be
#           adjusted within the program
#
#   Outputs: 
#       - u: the final results u(x) in [0,1]. As the following paper,
#           the global binary result can be available by threshholding u
#           by any constant alpha in (0,1):
#
#           Nikolova, M.; Esedoglu, S.; Chan, T. F. 
#           Algorithms for Finding Global Minimizers of Image Segmentation and Denoising Models 
#           SIAM J. App. Math., 2006, 66, 1632-1648
#
#       - erriter: it returns the error evaluation of each iteration,
#           i.e. it shows the convergence rate. One can check the algorithm
#           performance.
#
#       - i: gives the total number of iterations, when the algorithm converges.
#
#       - timet: gives the total computation time.
#       
#   Example:
#       >> [u, erriter, i, timet] = CMF_Cut;
#
#       >> us = max(u, beta);  # where beta in (0,1)
#
#       >> imagesc(us), colormap gray, axis image, axis off;figure(gcf)
#
#       >> figure, loglog(erriter,'DisplayName','erriterN');figure(gcf)
#
#
#           
#   The original algorithm was proposed in the following papers:
#
#   [1] Yuan, J.; Bae, E.;  Tai, X.-C. 
#       A Study on Continuous Max-Flow and Min-Cut Approaches 
#       CVPR, 2010
#
#   [2] Yuan, J.; Bae, E.; Tai, X.-C.; Boycov, Y.
#       A study on continuous max-flow and min-cut approaches. Part I: Binary labeling
#       UCLA CAM, Technical Report 10-61, 2010
#
#   The mimetic finite-difference discretization method was proposed for 
#   the total-variation function in the paper:
#
#   [1] Yuan, J.; Schn{\"o}rr, C.; Steidl, G.
#       Simultaneous Optical Flow Estimation and Decomposition
#       SIAM J.~Scientific Computing, 2007, vol. 29, page 2283-2304, number 6
#
#   This software can be used only for research purposes, you should cite ALL of
#   the aforementioned papers in any resulting publication.
#
#   Please email cn.yuanjing@gmail.com for any questions, suggestions and bug reports
#
#   The Software is provided "as is", without warranty of any kind.
#
#
#                       Version 1.0
#           https://sites.google.com/site/wwwjingyuan/       
#
#           Copyright 2011 Jing Yuan (cn.yuanjing@gmail.com)      
#

def cmf_cut(ur, alpha):
    
    [rows, cols] = np.shape(ur)
    imgSize = rows*cols

    cc = 0.3         # step-size of the augmented Lagrangian method, typically [0.3, 3]
    errbound = 1e-4  # error bound for convergence
    numIter = 100    # maximum iteration number (100)
    steps = 0.16     # step-size for gradient projection step to total variation, typically [0.1, 0.17]

    # build up the data terms
    urb = 0.8*ur + 0.1
    Cs = -np.log(1.0 - urb).astype(np.float32)
    Ct = -np.log(urb).astype(np.float32)

    # the initial value of u is set to be an initial cut, see below.
    u = np.where(Cs > Ct, 1.0, 0.0).astype(np.float32)
    # the initial values of two terminal flows ps and pt are set to be the specified legal flows.
    ps = np.minimum(Cs, Ct)
    pt = ps

    # the initial value of the spatial flow fiels p = (pp1, pp2) is set to be zero.
    pp1 = np.zeros((rows, cols+1), dtype=np.float32)
    pp2 = np.zeros((rows+1, cols), dtype=np.float32)
    divp = pp1[:,1:cols+1] - pp1[:,0:cols] + pp2[1:rows+1,:] - pp2[0:rows,:]

    erriter = np.zeros((numIter, 1))

    for i in range(numIter):
        # update the spatial flow field p = (pp1, pp2):
        #   the following steps are the gradient descent step with steps as the step-size.
        pts = divp - (ps - pt + u/cc)
        pp1[:,1:cols] = pp1[:,1:cols] + steps*(pts[:,1:cols] - pts[:,0:cols-1]) 
        pp2[1:rows,:] = pp2[1:rows,:] + steps*(pts[1:rows,:] - pts[0:rows-1,:])

        # the following steps give the projection to make |p(x)| <= alpha(x)
        gk = np.sqrt((pp1[:,0:cols]**2 + pp1[:,1:cols+1]**2 + pp2[0:rows,:]**2 + pp2[1:rows+1,:]**2)*0.5)
        gk = (np.where(gk <= alpha, 1, 0) + np.where(gk <= alpha, 0, 1) * (gk / alpha)).astype(np.float32)
        gk = 1 / gk
        
        pp1[:,1:cols] = (0.5*(gk[:,1:cols] + gk[:,0:cols-1])) * pp1[:,1:cols] 
        pp2[1:rows,:] = (0.5*(gk[1:rows,:] + gk[0:rows-1,:])) * pp2[1:rows,:]
        divp = pp1[:,1:cols+1] - pp1[:,0:cols] + pp2[1:rows+1,:] - pp2[0:rows,:]
        
        # updata the source flow ps
        pts = divp + pt - u/cc + 1/cc
        ps = np.minimum(pts, Cs)
    
        # update the sink flow pt
        pts = - divp + ps + u/cc
        pt = np.minimum(pts, Ct)

	# update the multiplier u
        erru = cc*(divp + pt  - ps)
        u = u - erru

        # evaluate the avarage error
        erriter[i] = np.sum(np.abs(erru))/imgSize
        
        if erriter[i] < errbound:
            break

    print('number of iterations = %u' % (i+1))
    return u, erriter, i



def graphcut_segm(I, area, K, alpha, sigma):
    [ minx, miny, maxx, maxy ] = area
    [h,w,c] = np.shape(I)
    dw = maxx - minx + 1
    dh = maxy - miny + 1
    mask = np.zeros((h, w), dtype=np.int16)
    mask[miny:miny+dh, minx:minx+dw] = 1

    grey = 0.2989*I[:,:,0] + 0.5870*I[:,:,1] + 0.1140*I[:,:,2]

    blur = 0.5 # amount of blur to reduce noise
    gauss = np.array([[ math.exp(-(i*i)/(2*blur*blur)) for i in range(-3, 4) ]], dtype=np.float32)
    gauss = gauss/np.sum(gauss) # normalize to one
    grey = convolve2d(grey, gauss.T @ gauss, 'same', 'symm')

    sobel = np.array([[-1, 0, 1], [-2,0,2], [-1,0,1]], dtype=np.float32)/4
    dx = convolve2d(grey, sobel,   'same', 'fill', 0)
    dy = convolve2d(grey, sobel.T, 'same', 'fill', 0)
    grad = np.sqrt(dx**2 + dy**2)

    edge = (alpha*sigma)*np.ones((h,w)) / (grad + sigma)

    for l in range(3):
        print('Find Gaussian mixture models...')
        fprob = mixture_prob(I, K, 10, mask)
        bprob = mixture_prob(I, K, 10, 1-mask)
        prior = np.reshape(fprob/(fprob + bprob), (h, w))

        print('Find minimum cut...')
        [u, erriter, i] = cmf_cut(prior, edge)
        mask = (u>0.5).astype(np.int16)

    return mask, prior

##############################


def graphcut_example():
    scale_factor = 0.5           # image downscale factor
    area = [ 120, 50, 220, 100 ] # image region to train foreground with [ minx, miny, maxx, maxy ]
    K = 8           #15        # number of mixture components
    alpha = 6.0      #8            # maximum edge cost
    sigma = 25.0     #20            # edge cost decay factor

    #img = Image.open('Images-jpg/tiger3.jpg')
    img = Image.open('data/a1_1.png')
    img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))

    area = [ int(i*scale_factor) for i in area ]
    I = np.asarray(img).astype(np.float32)
    segm, prior = graphcut_segm(I, area, K, alpha, sigma)
    
    Inew = mean_segments(img, segm)
    if True:
        Inew = overlay_bounds(img, segm)

    img = Image.fromarray(Inew.astype(np.ubyte))
    
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    a1 = f.add_subplot(2, 1, 1)
    plt.imshow(img)
    a1.title.set_text("")
    a1 = f.add_subplot(2, 1, 2)
    plt.imshow(prior)#np.array(I.astype('uint8')))
    a1.title.set_text("")
    plt.show()

if __name__ == '__main__':
    sys.exit(graphcut_example())
