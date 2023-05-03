import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D




def main():
    # find path to all images in folder
    folder_path = "data/"
    file_paths = find_files(folder_path)
    file_paths = [folder_path + file_path for file_path in file_paths]
    # load all images in folder 
    images = [cv2.imread(file_path) for file_path in file_paths]
    # Find ellipses for all images
    for image in images:
        ellipse = find_object(image)
        cv2.ellipse(image, ellipse, (0,255,0), 2)    
        # plot new edges
        cv2.imshow("new edges", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    pass

def find_files(folder_path):
    """
    This function finds all files in a given folder path
    """
    import os
    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            files.append(file)
    return files




def find_object(image):
    # Read image using cv2.imread()
    #path = "data/sr_2.png"
    img = image.copy()
    # gaussian blur
    img = cv2.GaussianBlur(img, (7,7), 8)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_b = img[:,:,0]
    img_g = img[:,:,1]
    img_r = img[:,:,2]
    img_s = img_hsv[:,:,1]
    img_v = img_hsv[:,:,2]

    # Calc the derivative in x and y direction for each img_gray, img_b, img_g, img_r, img_s, img_v
    img_gray = enhance(img_gray)
    img_b = enhance(img_b)
    img_g = enhance(img_g)
    img_r = enhance(img_r)
    img_s = enhance(img_s)
    img_v = enhance(img_v)

    img[:,:,0] = img_b
    img[:,:,1] = img_g
    img[:,:,2] = img_r
        
    # take greyscale image and apply thresholding
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # convolution with gaussian filter
    img = cv2.GaussianBlur(img, (7,7), 1)    

    # find edges in image 
    edges = cv2.Canny(img, 58, 280) # 20, 300)

    # dilate edges to make them thicker
    #kernel = np.ones((10,10),np.uint8)
    #edges = cv2.dilate(edges,kernel,iterations = 1)

    # perform closing 
    kernel = np.ones((3,3),np.uint8)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    new_img = np.ones_like(img)*255
    
    # sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        # draw contour on image
        cv2.drawContours(img, [contour], 0, (0,255,0), 2)
        # fit min enclosing ellipse to contour
        ellipsis = cv2.fitEllipse(contour)
        #check area of ellipse
        are_ellipse = np.pi*ellipsis[1][0]*ellipsis[1][1]
        if are_ellipse < img.shape[0]*img.shape[1]/3:
            cv2.ellipse(new_img, ellipsis, (0,255,0), 2)
        circle = cv2.minEnclosingCircle(contour)
        cv2.circle(new_img, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), (0,255,0), 2)
        #square = cv2.minAreaRect(contour)        
        #cv2.drawContours(new_img, [np.int0(cv2.boxPoints(square))], 0, (0,255,0), 2)  

        
    
    new_img = 255-new_img
    # Inflate new image 
    kernel = np.ones((3,3),np.uint8)
    new_img = cv2.dilate(new_img,kernel,iterations = 1)
    
    
    # Find contours in new image
    contours, hierarchy = cv2.findContours(new_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Remove contours with area bigger than half of the image
    contours = [contour for contour in contours if cv2.contourArea(contour) < img.shape[0]*img.shape[1]/3]
        
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # draw contours on image
    #cv2.drawContours(img, contours, -1, (0,255,0), 2)

    # Get the largest contour
    largest_contour = contours[0]
    
    # Fit ellipse to largest contour
    ellipsis = cv2.fitEllipse(largest_contour)
    ##draw ellipse on image
    #cv2.ellipse(img, ellipsis, (0,255,0), 2)    
    ## Fit rectangle to largest contour
    #
    #
    ## plot new edges
    cv2.imshow("new edges", new_img)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return ellipsis

def enhance(img):
    """
    This function enhances the image by calculating the gradient of the image and adding it to the original image with smart scaling
    """
    # Convert image to grayscale
    gray = img.copy()

    # Apply the Sobel operator in both the x and y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude of the gradients
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the magnitude to the range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    # Add the magnitude to the original image with smart scaling
    enhanced_img = cv2.addWeighted(img, 1, magnitude, 1, 0)

    return enhanced_img


















main()