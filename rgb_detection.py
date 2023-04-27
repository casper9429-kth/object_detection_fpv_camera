import numpy as np
import cv2 # OpenCV
import scipy
import sklearn
import random as rn

def main():

    #frame = take_picture("/dev/video2",debug=False)
    #frame = frame[:380, :]
    # save frame as image file
    file_name = "data/rgb_image.png"

    #ellipse = rgb_dectection_img(frame,debug=True)
    








def rgb_dectection2(file_path,debug=False):
    # Load the image
    image = cv2.imread(filename=file_path) # B G R

    # Gaussian blur the image with a kernel size of 5 and a standard deviation of 10
    image = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=10, sigmaY=10)

    # Scale image so first dimension is 400 pixels
    scale = 400 / image.shape[0]
    image = cv2.resize(src=image, dsize=(0, 0), fx=scale, fy=scale)

    # Downsample the image by a factor of 10
    #image = cv2.resize(src=image, dsize=(0, 0), fx=0.1, fy=0.1)

    # 






def rgb_dectection_img(image,debug=False):

    # Transform image to float32
    image = image.astype(np.uint8)

    # Gaussian blur the image with a kernel size of 5 and a standard deviation of 10
    image = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=10, sigmaY=10)

    # Scale image so first dimension is 400 pixels
    scale = 400 / image.shape[0]
    image = cv2.resize(src=image, dsize=(0, 0), fx=scale, fy=scale)

    # Downsample the image by a factor of 10
    #image = cv2.resize(src=image, dsize=(0, 0), fx=0.1, fy=0.1)




    # Peform canny edge detection for each channel of the image
    edges_b = cv2.Canny(image=image[:, :, 0], threshold1=80, threshold2=300)
    edges_g = cv2.Canny(image=image[:, :, 1], threshold1=80, threshold2=300)
    edges_r = cv2.Canny(image=image[:, :, 2], threshold1=80, threshold2=300)
    edges_gray = cv2.Canny(image=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY), threshold1=120, threshold2=170)

    # Plot the edges for each channel of the image in differenct colors
    # draw edges


    # Find contours
    #contours_b, hierarchy_b = cv2.findContours(image=edges_b, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    #contours_g, hierarchy_g = cv2.findContours(image=edges_g, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    #contours_r, hierarchy_r = cv2.findContours(image=edges_r, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    #contours_gray, hierarchy_gray = cv2.findContours(image=edges_gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    contours_b, hierarchy_b = cv2.findContours(image=edges_b,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    contours_g, hierarchy_g = cv2.findContours(image=edges_g,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    contours_r, hierarchy_r = cv2.findContours(image=edges_r,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    contours_gray, hierarchy_gray = cv2.findContours(image=edges_gray,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)


    # Merge all contours into one contours, and draw it
    # Merge all contours

    contours = contours_b + contours_g + contours_r + contours_gray

    # Transform the boundaries of the contours to one contour
            
    # Assume 'contours' is the list of contours obtained from your previous code

    # Create a blank image to draw the contours
    blank_image = np.zeros_like(image)

    # Draw all the contours on the blank image
    for contour in contours:
        cv2.drawContours(image=blank_image, contours=[contour], contourIdx=-1, color=(255, 255, 255), thickness=40)

    # Plot the image with all the contours drawn on it
    #if debug:
    #    cv2.imshow("All Contours", blank_image)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()



    # Find the outer contour of the combined contours
    contour_of_contours, _ = cv2.findContours(image=cv2.cvtColor(src=blank_image, code=cv2.COLOR_BGR2GRAY), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    

    # Fit ellipses to the outer contour of the combined contours
    if len(contour_of_contours) == 0:
        return None

    # sort contours by area
    contour_of_contours = sorted(contour_of_contours, key=cv2.contourArea, reverse=True)

    ellipsis = cv2.fitEllipse(contour_of_contours[0])
    
    if debug:
    

        ## Draw the contour of the contours on the original image (optional)
        result_image = image.copy()
        cv2.drawContours(image=result_image, contours=contour_of_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        cv2.ellipse(result_image, ellipsis, (0, 0, 255), 2)

        # Display the result image with the contour of the contours (optional)
        cv2.imshow('Result Image with Contour of Contours', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    


    return ellipsis



def rgb_dectection(file_path,debug=False):
    # Load the image
    image = cv2.imread(filename=file_path) # B G R

    # Gaussian blur the image with a kernel size of 5 and a standard deviation of 10
    image = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=10, sigmaY=10)

    # Scale image so first dimension is 400 pixels
    scale = 400 / image.shape[0]
    image = cv2.resize(src=image, dsize=(0, 0), fx=scale, fy=scale)

    # Downsample the image by a factor of 10
    #image = cv2.resize(src=image, dsize=(0, 0), fx=0.1, fy=0.1)




    # Peform canny edge detection for each channel of the image
    edges_b = cv2.Canny(image=image[:, :, 0], threshold1=200, threshold2=700)
    edges_g = cv2.Canny(image=image[:, :, 1], threshold1=200, threshold2=700)
    edges_r = cv2.Canny(image=image[:, :, 2], threshold1=200, threshold2=700)
    edges_gray = cv2.Canny(image=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY), threshold1=200, threshold2=700)

    # Find contours
    #contours_b, hierarchy_b = cv2.findContours(image=edges_b, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    #contours_g, hierarchy_g = cv2.findContours(image=edges_g, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    #contours_r, hierarchy_r = cv2.findContours(image=edges_r, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    #contours_gray, hierarchy_gray = cv2.findContours(image=edges_gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    contours_b, hierarchy_b = cv2.findContours(image=edges_b,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    contours_g, hierarchy_g = cv2.findContours(image=edges_g,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    contours_r, hierarchy_r = cv2.findContours(image=edges_r,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    contours_gray, hierarchy_gray = cv2.findContours(image=edges_gray,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)


    # Merge all contours into one contours, and draw it
    # Merge all contours

    contours = contours_b + contours_g + contours_r + contours_gray

    # Transform the boundaries of the contours to one contour
            
    # Assume 'contours' is the list of contours obtained from your previous code

    # Create a blank image to draw the contours
    blank_image = np.zeros_like(image)

    # Draw all the contours on the blank image
    for contour in contours:
        cv2.drawContours(image=blank_image, contours=[contour], contourIdx=-1, color=(255, 255, 255), thickness=10)



    # Find the outer contour of the combined contours
    contour_of_contours, _ = cv2.findContours(image=cv2.cvtColor(src=blank_image, code=cv2.COLOR_BGR2GRAY), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # Fit ellipses to the outer contour of the combined contours
    if len(contour_of_contours) > 0:
        return None

    ellipsis = cv2.fitEllipse(contour_of_contours[0])
    
    if debug:
        cv2.ellipse(image, ellipsis, (0, 0, 255), 2)


        ## Draw the contour of the contours on the original image (optional)
        result_image = image.copy()
        cv2.drawContours(image=result_image, contours=contour_of_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)

        # Display the result image with the contour of the contours (optional)
        cv2.imshow('Result Image with Contour of Contours', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    


    return ellipsis

# Function to take a picture from the camera
def take_picture(camera_device="/dev/video0",debug=False):
    """
    Take picture from the camera
    To determine the camera device, run the following command in the terminal:
    v4l2-ctl --list-devices
    This command will list all the devices connected to the computer.
    For laptop, the camera device is usually /dev/video0
    For the NUC, the arm camera device is usually /dev/video6
    Will return the image as a numpy array in BGR format uint8
    """
    # v4l2-ctl --list-devices
    cap = cv2.VideoCapture(camera_device)
    ret, frame = cap.read()
    if debug:
        cv2.imwrite("image.png", frame)
    return frame



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



if __name__ == '__main__':
    main()