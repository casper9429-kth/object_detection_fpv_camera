import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
import random as rn
from collections import defaultdict
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import copy 
import colorsys
import numpy.linalg as la
from collections import Counter
#from scipy.ndimage.filters import gaussian_filter
import random as rnd
from scipy.stats import multivariate_normal
# Do it fast and simple, no time.
# Plan first, code later
from scipy.ndimage import gaussian_filter
def main():
    img_color = np.array(Image.open('Images-jpg/orange.jpg'))#       .convert('L')) #you can pass multiple arguments in single line
    img_color = img_color.astype(np.float64)
    
    seg,center = kmeans_segm(img_color,50,50)
    
    new_image = center[seg[:,:]] 


    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    a1 = f.add_subplot(2, 1, 1)
    plt.imshow(new_image)
    a1.title.set_text("")
    a1 = f.add_subplot(2, 1, 2)
    plt.imshow(img_color)
    a1.title.set_text("")
	

        
    plt.show()
    print("lol")
    

    
def kmeans_segm_test():
    #img_color = np.array(Image.open('Images-jpg/orange.jpg'))#       .convert('L')) #you can pass multiple arguments in single line
    #img_grey = np.array(Image.open('Images-jpg/tiger1.jpg').convert('L'))
    #image = np.load("Images-npy/few256.npy")
    
    ###   
    K = 7              # number of clusters used
    L = 10              # number of iterations
    seed = 11           # seed used for random initialization
    scale_factor = 0.5  # image downscale factor
    image_sigma = 1   # image preblurring scale
    
    # conv 
    
    img = Image.open('Images-jpg/orange.jpg')
    img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))
    
    
    h = ImageFilter.GaussianBlur(image_sigma)
    #I = np.asarray(img.filter(ImageFilter.GaussianBlur(image_sigma))).astype(np.float32)
    I = np.asarray(img).astype(np.float64)    
    I =     gaussian_filter(I, sigma=image_sigma)

    #I = np.array(I.astype('uint8'))
    ###
        
    seg,center = kmeans_segm(I,K,L,seed)
    new_image = center[seg[:,:]]
    old_image =  np.array(I.astype('uint8'))
    # take difference
    #diff = old_image-new_image
    #diff = np.einsum('ijk,ijk->ij',diff,diff)
    #diff = np.sum(np.sqrt(diff))
    #print(diff)
    
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    a1 = f.add_subplot(2, 1, 1)
    plt.imshow(new_image)
    a1.title.set_text("")
    a1 = f.add_subplot(2, 1, 2)
    plt.imshow(np.array(I.astype('uint8')))
    a1.title.set_text("")
    plt.show()
    


def kmeans_segm_old(image, K, L, seed = 42):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """ 
    print("K: " + str(K))
    
    #settings
    smart_random = True
    
    
    # Set random seed
    rn.seed(seed)
        
    # Random initialization of K centers   
    if smart_random == True:
        k_centers = find_random_color_from_img(image,K,seed)
        k_centers = np.array(k_centers.astype('uint8'))
        
    else:
        max_val = 255
        k_centers = np.array([[rn.random()*max_val,rn.random()*max_val,rn.random()*max_val] for i in range(K)  ])
        k_centers = np.array(k_centers.astype('uint8'))


    shape = np.shape(image)
    img_extend = np.ones([shape[0],shape[1],len(k_centers),3])
    img_extend = np.einsum('ijk,ijnk->ijnk',image,img_extend)
        

    # Iter L times    
    for l in range(L):
        
        print("iter " + str(l) + " out of " + str(L) )
        # Compute distance between clusters and pixels
        # cluster pixel ij to cluster k
        #dist_img_cen = image[:,:]-k_centers[:]
        #scipy.spatial.distance_matrix(image, k_centers)
        
        # Alloc array
        # create image with each vector as it self [x,y,k replicas of it self]
        #shape = np.shape(image)
        #img_extend = np.ones([shape[0],shape[1],len(k_centers),3])
        #img_extend = np.einsum('ijk,ijnk->ijnk',image,img_extend)
        
        
        # Test case
        #a = img_extend[:,:,4,:]
        #b  = image     
        #print((a == b).all() )

        # subtract k_centers from img _extended
        dist_diff = img_extend[:,:] - k_centers
        
        # now calc length 
        dist = np.einsum('ijkv,ijkv->ijk',dist_diff,dist_diff)
                    
        #dist = image[:,:,None] - k_centers[:,None]  
        #dist = np.einsum('ijv,kv->ijk',image,k_centers)
        
        
        
        
        ## Assign each pixel to cluster centers with lowest distance
        # (ijk)                  ij pixel belongs to cluster k
        pixel_cluster_map  = np.argmin(dist, axis=2)#np.argmin(a, axis=None, out=None, *, keepdims=<no value>)
        
        ## Recompute all cluster centers by taking mean of the pixels assigned to each cluster
        # i cluster, x,y for pixel in org img        
        index_list_each_cluster = [np.argwhere(pixel_cluster_map == i) for i in range(K)]    
        for c in range(K):
            if len(index_list_each_cluster[c]) == 0:
                #k_centers[c] = [rn.random()*max_val,rn.random()*max_val,rn.random()*max_val]
                continue
            sum_r = 0
            sum_g = 0
            sum_b = 0
            for x,y in index_list_each_cluster[c]:
                sum_r += image[x,y,0]
                sum_g += image[x,y,1]
                sum_b += image[x,y,2] 
                
            sum_r = sum_r/len(index_list_each_cluster[c])
            sum_g = sum_g/len(index_list_each_cluster[c])
            sum_b = sum_b/len(index_list_each_cluster[c])
            k_centers[c] = [sum_r,sum_g,sum_b]
        
    
    
    
    centers = k_centers
    segmentation = pixel_cluster_map
    
    return segmentation, centers


def kmeans_segm(image, K, L, seed = 42):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """ 
    print("K: " + str(K))
    
    #settings
    smart_random = True
    
    
    # Set random seed
    rn.seed(seed)
        
    # Random initialization of K centers   
    if smart_random == True:
        k_centers = find_random_color_from_img(image,K,seed)
        k_centers = np.array(k_centers.astype('uint8'))
        
    else:
        max_val = 255
        k_centers = np.array([[rn.random()*max_val,rn.random()*max_val,rn.random()*max_val] for i in range(K)  ])
        k_centers = np.array(k_centers.astype('uint8'))


    shape = np.shape(image)
    img_extend = np.ones([shape[0],shape[1],len(k_centers),3])
    img_extend = np.einsum('ijk,ijnk->ijnk',image,img_extend)
        

    # Iter L times    
    for l in range(L):
        
        print("iter " + str(l) + " out of " + str(L) )

        # subtract k_centers from img _extended
        dist_diff = img_extend[:,:] - k_centers
        
        # now calc length 
        dist = np.einsum('ijkv,ijkv->ijk',dist_diff,dist_diff)
        
        ## Assign each pixel to cluster centers with lowest distance
        # (ijk)                  ij pixel belongs to cluster k
        pixel_cluster_map  = np.argmin(dist, axis=2)#np.argmin(a, axis=None, out=None, *, keepdims=<no value>)
        
        ## Recompute all cluster centers by taking mean of the pixels assigned to each cluster
        # i cluster, x,y for pixel in org img
        
        
        
        #image_k = image[pixel_cluster_map==]        
        #index_list_each_cluster = [np.argwhere(pixel_cluster_map == i) for i in range(K)]    
        for c in range(K):
            color_sum = image[pixel_cluster_map==c]
            k_centers[c] = np.einsum('ij->j',color_sum)/color_sum.shape[0]

        
    
    
    
    centers = k_centers
    segmentation = pixel_cluster_map
    
    return segmentation, centers

def find_random_color_from_img(image,K,seed):
    """finds K unique colors in an image"""
    rn.seed(seed)
    
    # Create color mat for image
    image_int = np.array(image)
    image_flat = np.reshape(image_int, (-1, 3)).astype(np.float64)
    
    rgb_unique= list(set(tuple(rgb) for rgb in image_flat))
    
    ret_var = np.array([rgb_unique[rn.randrange(0,len(rgb_unique))] for i in range(K)])
    return ret_var


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used 
    Output:
        prob: an image with probabilities per pixel
    """ 
    # some start var
    sigma_k_start = 100
    seed = 0
    std = 1
    
    # blur image to make kmeans work better
    image = gaussian_filter(image, sigma=std)
    
    # flat image
    image_flat = np.reshape(image, (-1, 3)).astype(np.float64)
    
    
    # for k means clustering position doesn't matter and for statistical properties it is the same. Therefore we disregard the form
    I = image[mask == 1] # masked part of image

    
    # we use kmeans_segm to get the color clusters, size of each cluster.
    # we therefore fold up I in an arbitrary way so that kmeans_cluster can handle it
    fac1,fac2 = find_big_factos(len(I))
    I_folded_up =  np.reshape(I, [fac1,fac2,3]).astype(np.float64)
    print("initialization of mu,sigma etc started")
    print("kmeans started")
    
    seg, cent = kmeans_segm(I_folded_up,K,L,seed)
    print("kmeans done")
    
    # We now have the data needed to calc mu_k, sigma_k and w_k
    
    mu_k = cent # eacg cluster mean color
    
    sigma_k  = np.zeros([K,3,3])#ones(K)*100 # k ident matrices 
    sigma_k[:]= np.identity(3)*sigma_k_start
    
    w_k = find_w_k(seg,cent,K) # function to find inital w_k
    
    # find clusters that correspond to the masked area of the image
    # We use one for loop to iterate L times over I(masked image)
    for l in range(L):
        # step 1: 
        # given mu_k,sigma_k,w_k calc p_ik = prob that pixel i belongs to cluster k
        p_ik = calc_p_ik(I,K,w_k,mu_k,sigma_k)
                
        # step 2:
        # update mu_k, sigma_k, w_k

        w_k = update_w_k(p_ik)
        
        mu_k = update_mu_k(p_ik,I)
        
        sigma_k = update_sigma_k(p_ik,I,mu_k,K)
                
        
    # We know have k clusters with weigths, sigma and mu trained on the object in image
    # next step is to check pixels in entire image and check the prob that each pixel belong to any of these clusters    
    p_i = calc_p_i(image_flat,K,w_k,mu_k,sigma_k)
    
    
    
    return p_i





def update_w_k(p_ik):
    """
    calc w_k bt summing the prob of all pixels for each cluster. Then dividing by the amount of pixels
    """
    shape = np.shape(p_ik)
    w_k = (1/shape[0])*np.einsum('ik->k',p_ik)    
    return w_k

def update_mu_k(p_ik,image_flat):
    """
    multiply each pixel with its proability of belonging to each cluster for every cluster, 
    then summing up the weigthed colours for each cluster to get the mean colour of each cluster
    """
    #  multiply each pixel with its proability of belonging to each cluster for every cluster and summing up the weigthed colours for each cluster
    sum_pik_ci = np.einsum('ik,ic->kc',p_ik,image_flat)
    #sum the prob that an arbitrary pixel belongs to each cluster
    sum_pik = np.einsum('ik->k',p_ik)
    # normalize the weigthed mean colour
    mu_k = sum_pik_ci[:]/sum_pik[:,None]
    return mu_k

def update_sigma_k(p_ik,image_flat,mu_k,K):
    """
    calc cov mat for each cluster
    """
    # get the diff between the each pixel and each clusters mean colour
    v = (image_flat[:,None] - mu_k[None,:])
    # create a cov matrix for each cluster which is the sum of cov matrix for each pixel
    sigma_talj = np.einsum('ik,ikj,ikm->kjm',p_ik,v,v) # swap dim
    # normalize factor is the sum of the prob that a pixel belongs to cluster k
    sum_pik = np.einsum('ik->k',p_ik)
    
    # do the normalization
    # to avoid zeros add something little
    sigma_k = sigma_talj[:]/sum_pik[:,None,None]
    sigma_k = sigma_k[:] + np.eye(3)*1e-3 
    
    return sigma_k

def calc_g_ki(image_flat,K,w_k,mu_k,sigma_k):
    """calc the multinomial prob density function value for each pixel to belong to each cluster"""

    ### create factor 
    g_k = 1/np.sqrt(((2*np.pi)**3)*la.det(sigma_k[:])) # works

    ### create exponent 
    v = ((image_flat[:,None,:] - mu_k[None,:,:])) # c - mu
    sigma_inv = la.inv(sigma_k[:])                # inv(cov)
    expon = -0.5*np.einsum('abi,bik,abk->ab',v,sigma_inv,v,optimize='greedy') # -1/2 (c-mu).T @ inv(cov) @ (c-mu)
    expon = np.exp(expon) # exp(-1/2 (c-mu).T @ inv(cov) @ (c-mu))
    
    g_k_i = expon[:,:]*g_k[None,:] # 1/sqrt(2pi**3 det(cov)) *  exp(-1/2 (c-mu).T @ inv(cov) @ (c-mu))
    return g_k_i
    
def calc_p_ik(image_flat,K,w_k,mu_k,sigma_k): # correct
    """
    Calc probabilities for each pixel p_i
    p_i = sum_k->K w_k*g_k(c_i) return one probability for each pixel in image
    """
    
    ## calc g_k_i, see func for details
    g_k_i = calc_g_ki(image_flat,K,w_k,mu_k,sigma_k)
    
    ## From g_k_i we can calc wg_k_i
    wg_k_i = w_k[None,:]*g_k_i[:,:]
    
    ## We have everything we need to calc p_i
    p_i = np.einsum('ij->i', wg_k_i)
    
    
    p_ik = wg_k_i[:,:]/p_i[:,None]
    
    return p_ik

def calc_p_i(image_flat,K,w_k,mu_k,sigma_k):
    """
    Calc probabilities for each pixel p_i
    p_i = sum_k->K w_k*g_k(c_i) return one probability for each pixel in image
    """
    ## calc g_k_i, see func for details
    g_k_i = calc_g_ki(image_flat,K,w_k,mu_k,sigma_k)

    ## From g_k_i we can calc wg_k_i
    wg_k_i = w_k[None,:]*g_k_i[:,:]
    
    ## We have everything we need to calc p_i
    p_i = np.einsum('ij->i', wg_k_i)
    
    return p_i
    
def find_w_k(seg,cent,K):
    """function to find inital weigths"""
    
    # add upp elements that contains the index of each cluster
    index_list_each_cluster = [np.argwhere(seg == i) for i in range(K)]
    w_k = [len(k) for k in index_list_each_cluster]
    # w_k now contains the number of pixels for each k. All pixels belong to an K therefore the sum of w_k is the total number of pixels
    sum = np.einsum('i->',w_k)
    # normalize w_k
    w_k = w_k/sum
    return w_k

def find_big_factos(n):
    """Find the two largest factors in one number n"""
    
    sq = int(np.sqrt(n))
    factor1 = 0
    factor2 = 0 
    for i in range(0,-sq,-1):
        if (n%(sq+i))==0:
            factor1 = int(sq+i)
            factor2 = int(n/factor1)
            return factor1, factor2
        
    
    if factor1 == 0 or factor2 == 0:
        raise Exception("No factor found")
    
    return 0,0
            






















main()
#kmeans_segm_test()