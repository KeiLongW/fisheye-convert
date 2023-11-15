import cv2
import numpy as np
import yaml

# Load the YAML file
with open("/home/keilong/Documents/RTAB-Map/camera_info/230222110188_right.yaml", "r") as file:
    data = yaml.safe_load(file)

# Extract camera parameters
camera_matrix = np.array(data["camera_matrix"]["data"]).reshape((3, 3))
dist_coefficients = np.array(data["distortion_coefficients"]["data"])
rectification_matrix = np.array(data["rectification_matrix"]["data"]).reshape((3, 3))
projection_matrix = np.array(data["projection_matrix"]["data"]).reshape((3, 4))

# Read the fisheye image
fisheye_image = cv2.imread("/home/keilong/Desktop/realsense_testing_img_20231114/R8_Fisheye.png")

# Undistort the fisheye image
undistorted_image = cv2.fisheye.undistortImage(
    fisheye_image,
    camera_matrix,
    dist_coefficients,
    Knew=projection_matrix
)

# cv2.imwrite("/home/keilong/Desktop/realsense_testing_img/test2_Fisheye_rectified.png", undistorted_image)

# Display the original and undistorted images
cv2.imshow("Original Image", fisheye_image)
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()