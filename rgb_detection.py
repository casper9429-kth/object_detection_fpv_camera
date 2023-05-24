import numpy as np
import cv2 # OpenCV
import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    frame = take_picture("/dev/video0",debug=False)
    # Cut out unnecessary parts of the image
    frame = frame[:380, :]
    # Find object
    ellipse = find_object_morph(frame)
    #ellipse = find_object_derivatives(frame)
    #ellipse = find_object_rembg(frame)
    # Draw ellipse
    cv2.ellipse(frame, ellipse, (0,255,0), 2)
    # Show image
    cv2.imshow("new edges", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_nitristic():
    # Camera matrix
    camera_matrix = np.array([[517.03632655, 0.0, 312.03052029], [0.0, 516.70216219, 252.01727667], [0.0, 0.0, 1.0]])

    # Distortion coefficients
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # find path to all images in folder
    folder_path = "data/"
    file_paths = find_files(folder_path)
    file_paths = [folder_path + file_path for file_path in file_paths]
    # load all images in folder 
    images = [cv2.imread(file_path) for file_path in file_paths]
    for image in images:

        h,  w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow("image", image)
        cv2.imshow("undistorted image", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        



def test():
    # find path to all images in folder
    folder_path = "data/"
    file_paths = find_files(folder_path)
    file_paths = [folder_path + file_path for file_path in file_paths]
    # load all images in folder 
    images = [cv2.imread(file_path) for file_path in file_paths]
    # Find ellipses for all images
    for image in images:
        #ellipse_morph = find_object_morph_pre_filter(image)
        #ellipse_derivatives = find_object_derivatives(image)
        #ellipsis_rembg = find_object_rembg(image)
        #cv2.ellipse(image, ellipse_morph, (0,0,255), 2)    
        #cv2.ellipse(image, ellipse_derivatives, (255,0,0), 2)
        #cv2.ellipse(image, ellipsis_rembg, (0,255,0), 2)
        # plot new edges
        #cv2.imshow("new edges", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        #plot_colour_space(image)
        
        filter_image(image)
        
    
    
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

def plot_colour_space(image):
    """
    Plot the colour space of the image
    """
    # Flatten the image array
    
    # downsample the image
    image = cv2.pyrDown(image)
    image = cv2.pyrDown(image)
    
    flatten = image.reshape((-1,3))

    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract H, S, V channels
    h = hsv[:,:,0].flatten()
    s = hsv[:,:,1].flatten()
    v = hsv[:,:,2].flatten()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot the points with colors
    ax.scatter(h, s, v, c=flatten/255.0)

    # Set axis labels
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')

    # Show the plot
    plt.show()
    

def filter_image(image):
    # to HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    image_orig = image.copy()
    image = image.copy()
    image = image/255.0 
    image = image -0.5
    
    
    
    # Calc covariance matrix of image
    cov = np.cov(image.reshape(-1,3).T)
    # Multiply covT with cov
    cov = np.matmul(cov.T, cov)
    # Find eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov)
    # Normalize eigenvectors
    eigvecs = eigvecs/np.linalg.norm(eigvecs, axis=0)
    # transform image to new basis
    eigvecs_inv = np.linalg.inv(eigvecs)
    image = np.einsum('ij,xyj->xyi', eigvecs_inv,image)
    # make biggest element in each pixel 1 and smallest 0
    # mask = np.zeros_like(image)
    # mask_1 = (image[:,:,0] > image[:,:,1]) & (image[:,:,0] > image[:,:,2])
    # mask_2 = (image[:,:,1] > image[:,:,0]) & (image[:,:,1] > image[:,:,2])
    # mask_3 = (image[:,:,2] > image[:,:,0]) & (image[:,:,2] > image[:,:,1])
    # image[mask_1,0] = 1
    # image[mask_1,1] = 0
    # image[mask_1,2] = 0
    # image[mask_2,0] = 0
    # image[mask_2,1] = 1
    # image[mask_2,2] = 0
    # image[mask_3,0] = 0
    # image[mask_3,1] = 0
    # image[mask_3,2] = 1

    # normalize image to 0-1
    image = image - np.min(image)
    image = image/np.max(image)
    image = image*255.0
    image = image.astype(np.uint8)

    # adaoptive thresholding
    image = cv2.adaptiveThreshold(image[:,:,0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    

    # Function to take a picture from the camera
def find_object_morph(image):
    """
    Find object using morphological operations
    """
    img = image.copy()
    # Greyscale image
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold image using adaptive thresholding
    #thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.adaptiveThreshold(grey, 255, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY, 69, 5)
    
    # Morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    blob = cv2.morphologyEx(blob, cv2.MORPH_DILATE, kernel)

    # Find contours in image 
    contours, hierarchy = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Fit ellipse to largest contour 
    #ellipse = cv2.fitEllipse(contours[0])
    #return ellipse


    new_img = np.ones_like(grey)*255
    
    for contour in contours:
        # draw contour on image
        cv2.drawContours(new_img, [contour], 0, (0,255,0), 2)
        # fit min enclosing ellipse to contour
        ellipsis = cv2.fitEllipse(contour)
        #check area of ellipse
        are_ellipse = np.pi*ellipsis[1][0]*ellipsis[1][1]
        if are_ellipse < img.shape[0]*img.shape[1]:
            cv2.ellipse(new_img, ellipsis, (0,255,0), 2)
        #circle = cv2.minEnclosingCircle(contour)
        #cv2.circle(new_img, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), (0,255,0), 2)
        #square = cv2.minAreaRect(contour)        
        #box = cv2.boxPoints(square)
        
    
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
    #cv2.imshow("new edges", new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return ellipsis


# Function to take a picture from the camera
def find_object_morph_pre_filter(image):
    """
    Find object using morphological operations
    """
    # Pre-filter image
    # Transform image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for color
    #lower = np.array([0, 3, 50])
    #upper = np.array([179, 39, 185])
    lower = np.array([0, 0, 0])
    upper = np.array([179, 39, 185])


    # Remove all pixels that are not in the color range
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)
    # show mask
    #cv2.imshow("first", mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



    # Morphological operation
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    #blob = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #blob = cv2.morphologyEx(blob, cv2.MORPH_DILATE, kernel)
    #mask = blob

    # erode mask
    #kernel = np.ones((12,12),np.uint8)
    kernel = np.ones((19,19),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)
    
    # 

    # show mask
    #cv2.imshow("mask", mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # dilate mask 
    kernel = np.ones((20,20),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)


    # Morphological operation to close holes
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    # show mask
    #cv2.imshow("mask", mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    



    # Find contours in image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # get largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Transform contours to polygons
    #contours = [cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True) for contour in contours]
    
    # Get convex hull of contours
    #contours = [cv2.convexHull(contour) for contour in contours]

    # convex hulls to polygons
    polygons = [(contour.reshape(-1,2)) for contour in contours]
    # transform to list of tuples
    polygons = [tuple(map(tuple, polygon)) for polygon in polygons]
    
    
    
    master_polygon = find_boundary_polygon(polygons)
    # visualize master polygon on image
    mask = np.zeros_like(image)    
    cv2.polylines(mask, [np.array(master_polygon)], True, (0,255,0), 2)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    mask = mask[:,:,1]

    # dilate mask 
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)


    # Find contours in new image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # contours 
    # sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    
    contour = contours[0]



    # Fit ellipse to largest contour
    ellipsis = cv2.fitEllipse(contour)
    return ellipsis

def find_boundary_polygon(convex_hulls):
    boundary_vertices = []

    for convex_hull in convex_hulls:
        for vertex in convex_hull:
            if vertex not in boundary_vertices:
                boundary_vertices.append(vertex)
            else:
                boundary_vertices.remove(vertex)

    # Sort the boundary vertices in clockwise order (assuming 2D points)
    boundary_vertices.sort(key= lambda v: (np.arctan2(v[1], v[0]), v[0]**2 + v[1]**2))
    return boundary_vertices

# Function to take a picture from the camera
def find_object_morph_pre_filter_old(image):
    """
    Find object using morphological operations
    """
    # Pre-filter image
    # Transform image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for color
    #lower = np.array([0, 3, 50])
    #upper = np.array([179, 39, 185])
    lower = np.array([0, 3, 50])
    upper = np.array([179, 39, 185])


    # Remove all pixels that are not in the color range
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)

    # erode mask
    #kernel = np.ones((12,12),np.uint8)
    #mask = cv2.erode(mask,kernel,iterations = 1)


    

    # show mask
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save mask
    img = image.copy()
    #grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #grey = cv2.bitwise_and(grey, grey, mask=mask)
    grey = mask

    # Gaussian blur
    grey = cv2.GaussianBlur(grey, (3,3), 0)
    # Threshold image using adaptive thresholding
    #thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.adaptiveThreshold(grey, 255, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY, 69, 5)
    
    # Morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    blob = cv2.morphologyEx(blob, cv2.MORPH_DILATE, kernel)

    # Find contours in image 
    contours, hierarchy = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Fit ellipse to largest contour 
    #ellipse = cv2.fitEllipse(contours[0])
    #return ellipse


    new_img = np.ones_like(grey)*255
    
    for contour in contours:
        # draw contour on image
        cv2.drawContours(new_img, [contour], 0, (0,255,0), 2)
        # fit min enclosing ellipse to contour
        ellipsis = cv2.fitEllipse(contour)
        #check area of ellipse
        are_ellipse = np.pi*ellipsis[1][0]*ellipsis[1][1]
        if are_ellipse < img.shape[0]*img.shape[1]:
            cv2.ellipse(new_img, ellipsis, (0,255,0), 2)
        #circle = cv2.minEnclosingCircle(contour)
        #cv2.circle(new_img, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), (0,255,0), 2)
        #square = cv2.minAreaRect(contour)        
        #box = cv2.boxPoints(square)
        

    # show new image
    #cv2.imshow("new image", new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
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
    #cv2.imshow("new edges", new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return ellipsis


def find_object_rembg(image):
    """
    Finds object using rembg library, returns ellipse fitted to object
    """
    # remove background from image
    img_without_background = rembg.remove(image)
    # Make img_without_background greyscale
    grey = cv2.cvtColor(img_without_background, cv2.COLOR_BGR2GRAY)
    # make grey image binary
    #grey = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    grey[grey > 0] = 255
    
    # Find contours in image
    contours, hierarchy = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = contours[0]
    
    # Fit ellipse,circle,rectangle to contour
    ellipsis = cv2.fitEllipse(contour)

    return ellipsis


def find_object_derivatives(image):
    """
    Finds object using derivatives
    """
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




if __name__ == '__main__':
    #main()
    #test_nitristic()
    test()