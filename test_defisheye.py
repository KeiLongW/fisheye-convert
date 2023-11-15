import cv2
import numpy as np


def undistort_fisheye(image_path, K, D):
    # Load the fisheye image
    img = cv2.imread(image_path)

    # Get image dimensions
    h, w = img.shape[:2]

    # Define the camera matrix
    new_camera_matrix0, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    new_camera_matrix1, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    
    # Undistort the fisheye image
    undistorted_img_undis_K = cv2.undistort(img, K, D, None, K)
    undistorted_img_undis_0 = cv2.undistort(img, K, D, None, new_camera_matrix0)
    undistorted_img_undit_1 = cv2.undistort(img, K, D, None, new_camera_matrix1)
    undistorted_img_fish_undit_K = cv2.fisheye.undistortImage(img, K, D, Knew=K)
    undistorted_img_fish_undit_0 = cv2.fisheye.undistortImage(img, K, D, Knew=new_camera_matrix0)
    undistorted_img_fish_undit_1 = cv2.fisheye.undistortImage(img, K, D, Knew=new_camera_matrix1)

    # Crop the image to remove black borders
    # x, y, w, h = roi
    # undistorted_img = undistorted_img[y:y+h, x:x+w]

    # Save the undistorted image
    cv2.imshow('Original Fisheye Image', cv2.resize(img, (424, 400)))
    cv2.imshow('Rectified Image (undistort K)', cv2.resize(undistorted_img_undis_K, (424, 400)))
    cv2.imshow('Rectified Image (undistort 0)', cv2.resize(undistorted_img_undis_0, (424, 400)))
    cv2.imshow('Rectified Image (undistort 1)', cv2.resize(undistorted_img_undit_1, (424, 400)))
    cv2.imshow('Rectified Image (fisheye undistort K)', cv2.resize(undistorted_img_fish_undit_K, (424, 400)))
    cv2.imshow('Rectified Image (fisheye undistort 0)', cv2.resize(undistorted_img_fish_undit_0, (424, 400)))
    cv2.imshow('Rectified Image (fisheye undistort 1)', cv2.resize(undistorted_img_fish_undit_1, (424, 400)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":    
    # fx = 286.332214355469
    # fy = 286.06201171875
    # cx = 418.016998291016
    # cy = 383.746704101562
    
    # k1 = -0.0072045000270009
    # k2 = 0.0415875911712646
    # p1 = -0.0384871289134026
    # p2 = 0.00634303921833634
    # k3 = 0
    
    fx = 289.8846763939774
    fy = 387.06227498911073
    cx = 316.58338771731684
    cy = 241.86918974530997
    
    k1 = -0.28145280823630314
    k2 = 0.1091225266070853
    p1 = 0.00018240521552769957
    p2 = 6.785645205189456e-05
    k3 = -0.023250148744302514
  
    # Replace these values with your camera matrix (K) and distortion coefficients (D)
    # K = np.array([[fx, 0, cx],
    #               [0, fy, cy],
    #               [0, 0, 1]])

    # D = np.array([k1, k2, p1, p2, k3])
    # image_path = "/home/keilong/repositories/drone-racing-dataset/data/autonomous/flight-01a-ellipse/camera_flight-01a-ellipse/01298_1691756170626810.jpg"
    
    K = np.array([[2.8325647864248356e+02, 0, 4.1338173265438456e+02],
                  [0, 2.8114932686815399e+02, 3.9173699544455792e+02],
                  [0, 0, 1]])

    D = np.array([ -1.2297211856003311e-02, 5.5005189166060731e-02,
       -4.7602777750879892e-02, 8.4774962087919032e-03 ])
    image_path = "/home/keilong/Desktop/realsense_testing_img_20231114/R8_Fisheye.png"

    undistort_fisheye(image_path, K, D)