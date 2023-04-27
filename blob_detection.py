import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.mixture import GaussianMixture

def main2():
    # Read image using cv2.imread()
    path = "data/bb_1.png"
    img = cv2.imread(path)
    
    # Gaussian blur image aggressively
    img = cv2.GaussianBlur(img, (7,7), 8)    

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)








    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
        
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    result = img.copy()
    for cnt in contours:
        cv2.drawContours(result, [cnt], 0, (0, 255, 0), 2)

    # Show the images
    cv2.imshow("Original Image", img)
    cv2.imshow("Binary Image", thresh)
    cv2.imshow("Morphological Operations", closing)
    cv2.imshow("Detected Objects", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
    # Read image using cv2.imread()
    path = "data/a3_2.png"
    img = cv2.imread(path)
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
    edges = cv2.Canny(img, 20, 300)
    # add edges to original image
    cv2.imshow("edges",edges)

    # Fit rectangle to edges 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        el = cv2.fitEllipse(cnt)
        cv2.ellipse(img, el, (0,255,0), 2)
        
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Find contour in img 
    gray_contour = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    b_contour = cv2.findContours(img_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    g_contour = cv2.findContours(img_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    r_contour = cv2.findContours(img_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    s_contour = cv2.findContours(img_s, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    v_contour = cv2.findContours(img_v, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    # Adaptive thresholding on the images
    thres_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thres_b = cv2.adaptiveThreshold(img_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thres_g = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thres_r = cv2.adaptiveThreshold(img_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh_s = cv2.adaptiveThreshold(img_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh_v = cv2.adaptiveThreshold(img_v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresholds = [thres_gray, thres_b, thres_g, thres_r, thresh_s, thresh_v]

    for i,theres in enumerate(thresholds):
        # apply gaussian filter
        thresholds[i] = cv2.GaussianBlur(theres,(7,7),8)
    
    # add all images onto each other
    stacked = np.zeros_like(thres_gray)
    stacked = stacked + thres_gray/6
    stacked = stacked + thres_b/6
    stacked = stacked + thres_g/6
    stacked = stacked + thres_r/6
    stacked = stacked + thresh_s/6
    stacked = stacked + thresh_v/6

    # Plor stacked 
    #cv2.imshow("stacked",stacked)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #0
        



    morph_images = []
    for thresh in thresholds:
        # Apply morphological operations on the images
        kernel = np.ones((5, 5), np.uint8)  
            
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        morph_images.append(thresh)
        morph_images.append(closing)



    images = morph_images
    
    
    
    
    
    
    
    
    
    
    
    fig, axes = plt.subplots(6, 2, figsize=(8, 8))
    axes = axes.flatten()
    
    for img, ax in zip(images, axes):
        
        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

    
    






    
    pass

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