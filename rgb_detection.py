import numpy as np
import cv2 # OpenCV

def main():
    frame = take_picture("/dev/video0",debug=False)
    # Cut out unnecessary parts of the image
    frame = frame[:380, :]
    # Find object
    ellipse = find_object_morph_pre_filter(frame)
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
        ellipse_morph = find_object_morph_pre_filter(image)
        #ellipse_derivatives = find_object_derivatives(image)
        #ellipsis_rembg = find_object_rembg(image)
        cv2.ellipse(image, ellipse_morph, (0,0,255), 2)    
        #cv2.ellipse(image, ellipse_derivatives, (255,0,0), 2)
        #cv2.ellipse(image, ellipsis_rembg, (0,255,0), 2)
        # plot new edges
        cv2.imshow("new edges", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        #plot_colour_space(image)
        
        
    
    
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

    # erode mask
    #kernel = np.ones((12,12),np.uint8)
    kernel = np.ones((19,19),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)
    
    # dilate mask 
    kernel = np.ones((20,20),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    # Find contours in image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # get largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # convex hulls to polygons
    polygons = [(contour.reshape(-1,2)) for contour in contours]
    # transform to list of tuples
    polygons = [tuple(map(tuple, polygon)) for polygon in polygons]
    # find master polygon
    master_polygon = find_boundary_polygon(polygons)
    # visualize master polygon on image
    mask = np.zeros_like(image)    
    cv2.polylines(mask, [np.array(master_polygon)], True, (0,255,0), 2)
    mask = mask[:,:,1]

    # dilate mask 
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)


    # Find contours in new image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # contours 
    # sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if len(contours) == 0:
        return None
    
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