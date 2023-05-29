import numpy as np
import cv2 # OpenCV

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


# load camera matrix and distortion coefficients
# it is saved in a .npz file
file_name = "calibration.npz"
# newcameramtx
# roi
# tvecs
# dist
# mtx
# rvecs
with np.load(file_name) as data:
    newcameramtx = data["newcameramtx"] # optimal camera matrix
    roi = data["roi"] # region of interest
    tvecs = data["tvecs"]
    dist = data["dist"]
    mtx = data["mtx"]
    rvecs = data["rvecs"]


# load image
#file_name = "data/2019-05-15-15-53-00.png"
#image = cv2.imread(file_name)
image = take_picture(camera_device="/dev/video0",debug=False)

# undistort image
#image = cv2.undistort(image, mtx, dist, None, newcameramtx)
# undistort image
image = cv2.undistort(image, mtx, dist, None, newcameramtx)
#image = cv2.undistort(image, mtx, dist)

# take random point in image
h = image.shape[0]
w = image.shape[1]
x = np.random.randint(0, w)
y = np.random.randint(0, h)
z = 1.16 # meters

# calculate pose in world
x_world, y_world, z_world = np.matmul(np.linalg.inv(newcameramtx), np.array([x, y, z]))

print("x_world: ", x_world)
print("y_world: ", y_world)
print("z_world: ", z_world)
# plot undistorted image and point with circle 
cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
# also draw circle at the center of the image
cv2.circle(image, (int(w/2), int(h/2)), 5, (0, 0, 255), -1)
# draw line from center of image to random point
cv2.line(image, (int(w/2), int(h/2)), (x, y), (0, 0, 255), 2)
# draw triangle from center of image to random point
cv2.line(image, (int(w/2), int(h/2)), (x, y), (0, 0, 255), 2)
cv2.line(image, (int(w/2), int(h/2)), (int(w/2), y), (0, 0, 255), 2)
cv2.line(image, (int(w/2), y), (x, y), (0, 0, 255), 2)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

