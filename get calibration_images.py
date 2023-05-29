import numpy as np
import cv2 as cv
import glob
import time

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
    cap = cv.VideoCapture(camera_device)
    ret, frame = cap.read()
    if debug:
        cv.imwrite("image.png", frame)
    return frame


# corners
c1 = 8
c2 = 6


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((c1*c2,3), np.float32)
objp[:,:2] = np.mgrid[0:c1,0:c2].T.reshape(-1,2)*108 # 108 is the size of the square in mm
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('*.jpg')
images = []
test_images = []
n_images = 10
i = 0
while True:
    
    img = take_picture(camera_device="/dev/video0",debug=False)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (c1,c2), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        i += 1
        if len(images) < n_images:
            images.append(img)
            # save the image
            cv.imwrite("image" + str(i) + ".png", img)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
        else:
            test_images.append(img)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (c1,c2), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
        cv.destroyAllWindows()
        if len(images) + len(test_images) > n_images:
            break
    
    time.sleep(1)

# calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# get optimal camera matrix
img = test_images[0]
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# show the image
cv.imshow('calibresult.png', dst)
cv.waitKey(500)
cv.destroyAllWindows()
cv.imwrite('calibresult.png', dst)

# reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

# save the calibration
np.savez("calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, newcameramtx=newcameramtx, roi=roi)


# with known distance to image plane, calculate x, y, z in the world frame
# distance to image plane
d = 0.5 # meter
# pixel location
x = 0
y = 0
# calculate x, y, z
x = (x - newcameramtx[0,2]) * d / newcameramtx[0,0]
y = (y - newcameramtx[1,2]) * d / newcameramtx[1,1]
z = d
print("x: {}, y: {}, z: {}".format(x, y, z))

